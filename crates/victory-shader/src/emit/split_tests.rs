//! End-to-end tests of wave64 compute programs split into two 32-lane host
//! waves (`structure::structure_split`, `lower`'s `Split`): synthetic
//! 64-thread GFX10 programs whose lanes read each other across the halves,
//! lowered to DXIL, validated by `dxil.dll` when present, and run on the
//! hardware adapter and WARP. Each lane stores its results to the raw UAV,
//! which is checked against what the console's single wave64 computes.

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
/// Inline integer constant `n` (0 to 64) as a source operand.
const fn k(n: u16) -> u16 {
    128 + n
}
const VCC_LO: u16 = 106;
const VCC_HI: u16 = 107;

/// SOP1 `s_mov_b32 sdst, src0`.
const fn s_mov_b32(sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32
}

/// SMEM `s_load_dwordx4 s[dst:dst+3], s[base:base+1], 0`.
const fn s_load_dwordx4(dst: u8, base: u8) -> [u32; 2] {
    [
        0x3D << 26 | 0x02 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        125 << 25,
    ]
}

/// VOP1 `op vdst, src0`.
const fn vop1(op: u16, vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// VOP2 `op vdst, src0, vsrc1`.
const fn vop2(op: u8, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    (op as u32) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// VOPC `op vcc, src0, vsrc1`.
const fn vopc(op: u8, src0: u16, vsrc1: u8) -> u32 {
    0x3E << 25 | (op as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// VOP3 `op vdst, src0, src1, src2` (no modifiers).
const fn vop3(op: u16, vdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    [
        0x35 << 26 | (op as u32 & 0x3FF) << 16 | vdst as u32,
        (src0 as u32 & 0x1FF) | (src1 as u32 & 0x1FF) << 9 | (src2 as u32 & 0x1FF) << 18,
    ]
}

/// MUBUF `buffer_store_dword vdata, vaddr, s[srsrc:srsrc+3], 0 offen
/// offset:byte_off`.
const fn buffer_store_dword_offen(vdata: u8, vaddr: u8, srsrc: u8, byte_off: u16) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18 | 1 << 12 | (byte_off as u32 & 0xFFF),
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8 | vaddr as u32,
    ]
}

const fn vgpr(n: u8) -> u16 {
    256 + n as u16
}

/// A wave64 compute program of 64 threads (thread id x in v0) running
/// `body`, with the raw UAV's descriptor in s[8:11] and v1 = 4 · v0, made
/// in split form. Checks that the code decodes to `names` first.
fn build_split(body: &[u32], names: &[&str]) -> HostShader {
    let mut words = vec![s_mov_b32(5, 0x80)]; // s5 = 0: table address high
    words.extend_from_slice(&s_load_dwordx4(8, 4));
    words.push(vop2(0x1A, 1, k(2), 0)); // v_lshlrev_b32 v1, 2, v0
    words.extend_from_slice(body);
    words.push(S_ENDPGM);
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let decoded: Vec<&str> = crate::gfx10::decode(&code)
        .expect("decode")
        .iter()
        .filter_map(|i| i.name())
        .collect();
    for n in names {
        assert!(decoded.contains(n), "{n} not in {decoded:?}");
    }
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 16,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 4,
            slot: 1,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1, // user SGPRs s0..s4
            threads: [64, 1, 1],
        }),
        code,
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure_split(&p).expect("structure_split");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })
    .expect("lower");
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
    };
    assert_eq!(
        shader.wave_size(),
        Some(32),
        "split programs run on 32-lane waves"
    );
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    shader
}

/// Runs `shader` (one group) on the hardware adapter and WARP with a raw
/// UAV of `dwords`; `check(backend, uav)`.
fn run(shader: &HostShader, dwords: u32, mut check: impl FnMut(&str, &[u32])) {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        // Split programs declare 32-lane waves (WARP runs 4).
        let (min, max) = host.wave_lanes();
        if !(min..=max).contains(&32) {
            eprintln!("{label}: waves of {min} to {max} lanes; skipped");
            continue;
        }
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(dwords)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        check(label, &out[0]);
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

#[test]
fn split_ballot_and_readlane_cross_the_halves() {
    // Lanes below 40 set VCC; v_mbcnt counts the set lanes below each lane,
    // which needs both halves' ballots. v_readlane reads lane 50 (the high
    // half) into s12 for every lane.
    let mut body = vec![vopc(0xC4, k(40), 0)]; // v_cmp_gt_u32 vcc, 40, v0
    body.extend_from_slice(&vop3(0x365, 2, VCC_LO, k(0), 0)); // v_mbcnt_lo_u32_b32 v2, vcc_lo, 0
    body.extend_from_slice(&vop3(0x366, 2, VCC_HI, vgpr(2), 0)); // v_mbcnt_hi_u32_b32 v2, vcc_hi, v2
    body.extend_from_slice(&vop3(0x360, 12, vgpr(0), k(50), 0)); // v_readlane_b32 s12, v0, 50
    body.push(vop1(0x001, 3, 12)); // v_mov_b32 v3, s12
    body.extend_from_slice(&buffer_store_dword_offen(2, 1, 8, 0));
    body.extend_from_slice(&buffer_store_dword_offen(3, 1, 8, 256));
    let shader = build_split(
        &body,
        &[
            "v_cmp_gt_u32",
            "v_mbcnt_lo_u32_b32",
            "v_mbcnt_hi_u32_b32",
            "v_readlane_b32",
        ],
    );
    run(&shader, 128, |label, uav| {
        for lane in 0..64u32 {
            assert_eq!(
                uav[lane as usize],
                lane.min(40),
                "{label}: mbcnt of lane {lane}"
            );
            assert_eq!(
                uav[64 + lane as usize],
                50,
                "{label}: readlane in lane {lane}"
            );
        }
    });
}

#[test]
fn split_branch_on_exec_keeps_the_wave_together() {
    // Lanes below 20 take EXEC; the high half then has none, and the
    // console wave still runs the block (some lane is set) with v_readlane
    // of lane 5 inside it. Every lane then stores what the block left.
    let mut body = vec![vopc(0xC4, k(20), 0)]; // v_cmp_gt_u32 vcc, 20, v0
    body.push(vop1(0x001, 2, k(0))); // v_mov_b32 v2, 0
    body.push(0xBE8E_246A); // s_and_saveexec_b64 s[14:15], vcc
    body.push(0xBF88_0003); // s_cbranch_execz +3
    body.extend_from_slice(&vop3(0x360, 12, vgpr(0), k(5), 0)); // v_readlane_b32 s12, v0, 5
    body.push(vop1(0x001, 2, 12)); // v_mov_b32 v2, s12
    body.push(0x88FE_0E7E); // s_or_b64 exec, exec, s[14:15]
    body.extend_from_slice(&buffer_store_dword_offen(2, 1, 8, 0));
    let shader = build_split(
        &body,
        &[
            "s_and_saveexec_b64",
            "s_cbranch_execz",
            "v_readlane_b32",
            "s_or_b64",
        ],
    );
    run(&shader, 64, |label, uav| {
        for lane in 0..64u32 {
            let want = if lane < 20 { 5 } else { 0 };
            assert_eq!(uav[lane as usize], want, "{label}: lane {lane}");
        }
    });
}
