#[cfg(target_arch = "x86")]
use core_arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core_arch::x86_64::*;

use byteorder::{ByteOrder, LittleEndian};
use core::mem;
use packed_simd::{i32x4, FromBits, IntoBits};

use crate::AlignedWords8;
use crate::Block;
use crate::Hash;
use crate::Params;
use crate::StateWords;
use crate::BLOCKBYTES;
use crate::IV;
use crate::SIGMA;

#[inline(always)]
unsafe fn loadu(p: *const u32) -> __m512i {
    _mm512_loadu_si512(p as *const __m512i)
}

#[inline(always)]
unsafe fn loadu128(x0: u32, x1: u32, x2: u32, x3: u32) -> __m128i {
    _mm_set_epi32(x0 as i32, x1 as i32, x2 as i32, x3 as i32)
}

#[inline(always)]
unsafe fn add(a: __m512i, b: __m512i) -> __m512i {
    _mm512_add_epi32(a, b)
}

#[inline(always)]
unsafe fn xor(a: __m512i, b: __m512i) -> __m512i {
    _mm512_xor_si512(a, b)
}

#[inline(always)]
unsafe fn rotr16(x: __m512i) -> __m512i {
    _mm512_ror_epi32(x, 16)
}

#[inline(always)]
unsafe fn rotr12(x: __m512i) -> __m512i {
    _mm512_ror_epi32(x, 12)
}

#[inline(always)]
unsafe fn rotr8(x: __m512i) -> __m512i {
    _mm512_ror_epi32(x, 8)
}

#[inline(always)]
unsafe fn rotr7(x: __m512i) -> __m512i {
    _mm512_ror_epi32(x, 7)
}

#[inline(always)]
unsafe fn load_512_from_u32(x: u32) -> __m512i {
    _mm512_set1_epi32(x as i32)
}

#[inline(always)]
unsafe fn load_512_from_16xu32(
    x1: u32,
    x2: u32,
    x3: u32,
    x4: u32,
    x5: u32,
    x6: u32,
    x7: u32,
    x8: u32,
    x9: u32,
    x10: u32,
    x11: u32,
    x12: u32,
    x13: u32,
    x14: u32,
    x15: u32,
    x16: u32,
) -> __m512i {
    // NOTE: This order of arguments for _mm512_set_epi32 is the reverse of how the ints come out
    // when you transmute them back into an array of u32's.
    _mm512_set_epi32(
        x16 as i32, x15 as i32, x14 as i32, x13 as i32, x12 as i32, x11 as i32, x10 as i32,
        x9 as i32, x8 as i32, x7 as i32, x6 as i32, x5 as i32, x4 as i32, x3 as i32, x2 as i32,
        x1 as i32,
    )
}

macro_rules! g {
    ($v:expr, $a:expr, $b:expr, $c:expr, $d:expr, $x:expr, $y:expr) => {
        $v[$a] = add($v[$a], $v[$b]);
        $v[$a] = add($v[$a], $x);

        $v[$d] = xor($v[$d], $v[$a]);
        $v[$d] = rotr16($v[$d]);

        $v[$c] = add($v[$c], $v[$d]);

        $v[$b] = xor($v[$b], $v[$c]);
        $v[$b] = rotr12($v[$b]);

        $v[$a] = add($v[$a], $v[$b]);
        $v[$a] = add($v[$a], $y);

        $v[$d] = xor($v[$d], $v[$a]);
        $v[$d] = rotr8($v[$d]);

        $v[$c] = add($v[$c], $v[$d]);

        $v[$b] = xor($v[$b], $v[$c]);
        $v[$b] = rotr7($v[$b]);
    };
}

pub unsafe fn compress(h: &mut StateWords, msg: &Block, count: u64, lastblock: u32, lastnode: u32) {
    // Initialize the compression state.
    let mut v = [
        h[0],
        h[1],
        h[2],
        h[3],
        h[4],
        h[5],
        h[6],
        h[7],
        IV[0],
        IV[1],
        IV[2],
        IV[3],
        IV[4] ^ count as u32,
        IV[5] ^ (count >> 32) as u32,
        IV[6] ^ lastblock,
        IV[7] ^ lastnode,
    ];

    // Parse the message bytes as ints in little endian order.
    let msg_refs = array_refs!(msg, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4);
    let m = [
        LittleEndian::read_u32(msg_refs.0),
        LittleEndian::read_u32(msg_refs.1),
        LittleEndian::read_u32(msg_refs.2),
        LittleEndian::read_u32(msg_refs.3),
        LittleEndian::read_u32(msg_refs.4),
        LittleEndian::read_u32(msg_refs.5),
        LittleEndian::read_u32(msg_refs.6),
        LittleEndian::read_u32(msg_refs.7),
        LittleEndian::read_u32(msg_refs.8),
        LittleEndian::read_u32(msg_refs.9),
        LittleEndian::read_u32(msg_refs.10),
        LittleEndian::read_u32(msg_refs.11),
        LittleEndian::read_u32(msg_refs.12),
        LittleEndian::read_u32(msg_refs.13),
        LittleEndian::read_u32(msg_refs.14),
        LittleEndian::read_u32(msg_refs.15),
    ];

    let v_old = v.clone();
    round(0, &m, &mut v);
    round(1, &m, &mut v);
    round(2, &m, &mut v);
    round(3, &m, &mut v);
    round(4, &m, &mut v);
    round(5, &m, &mut v);
    round(6, &m, &mut v);
    round(7, &m, &mut v);
    round(8, &m, &mut v);
    round(9, &m, &mut v);

    assert_ne!(v_old, v);

    h[0] ^= v[0] ^ v[8];
    h[1] ^= v[1] ^ v[9];
    h[2] ^= v[2] ^ v[10];
    h[3] ^= v[3] ^ v[11];
    h[4] ^= v[4] ^ v[12];
    h[5] ^= v[5] ^ v[13];
    h[6] ^= v[6] ^ v[14];
    h[7] ^= v[7] ^ v[15];
}

// Single blake2s round
#[inline(always)]
unsafe fn round(r: usize, m: &[u32; 16], v: &mut [u32; 16]) {
    // this can replace round in portable.rs
    // Idea, use mm128i to do 4 operations at a time

    let mut v0123 = loadu128(v[0], v[1], v[2], v[3]);
    let mut v4567 = loadu128(v[4], v[5], v[6], v[7]);
    let mut v891011 = loadu128(v[8], v[9], v[10], v[11]);
    let mut v12131415 = loadu128(v[12], v[13], v[14], v[15]);

    let ms0246 = loadu128(
        m[SIGMA[r][0] as usize],
        m[SIGMA[r][2] as usize],
        m[SIGMA[r][4] as usize],
        m[SIGMA[r][6] as usize],
    );
    let ms1357 = loadu128(
        m[SIGMA[r][1] as usize],
        m[SIGMA[r][3] as usize],
        m[SIGMA[r][5] as usize],
        m[SIGMA[r][7] as usize],
    );
    let ms8101214 = loadu128(
        m[SIGMA[r][8] as usize],
        m[SIGMA[r][10] as usize],
        m[SIGMA[r][12] as usize],
        m[SIGMA[r][14] as usize],
    );
    let ms9111315 = loadu128(
        m[SIGMA[r][9] as usize],
        m[SIGMA[r][11] as usize],
        m[SIGMA[r][13] as usize],
        m[SIGMA[r][15] as usize],
    );

    // Mix columns

    // add
    v0123 = _mm_add_epi32(v0123, ms0246);

    // add
    v0123 = _mm_add_epi32(v0123, v4567);

    // xor
    v12131415 = _mm_xor_si128(v12131415, v0123);

    // ror16
    v12131415 = _mm_ror_epi32(v12131415, 16);

    // add
    v891011 = _mm_add_epi32(v891011, v12131415);

    // add
    v4567 = _mm_xor_si128(v4567, v891011);

    // ror12
    v4567 = _mm_ror_epi32(v4567, 12);

    // add
    v0123 = _mm_add_epi32(v0123, ms1357);

    // add
    v0123 = _mm_add_epi32(v0123, v4567);

    // xor
    v12131415 = _mm_xor_si128(v12131415, v0123);

    // ror8
    v12131415 = _mm_ror_epi32(v12131415, 8);

    // add
    v891011 = _mm_add_epi32(v891011, v12131415);

    // xor
    v4567 = _mm_xor_si128(v4567, v891011);

    // ror7
    v4567 = _mm_ror_epi32(v4567, 7);

    // Mix rows
    // FIXME: correct shuffle masks
    let mut v5674 = _mm_shuffle_epi32(v4567, 0b0010_0001_0000_0011);
    let mut v15121314 = _mm_shuffle_epi32(v12131415, 0b0010_0001_0000_0011);
    let mut v101189 = _mm_shuffle_epi32(v891011, 0b0010_0001_0000_0011);

    // add
    v0123 = _mm_add_epi32(v0123, ms8101214);

    // add
    v0123 = _mm_add_epi32(v0123, v5674);

    // xor
    v15121314 = _mm_xor_si128(v15121314, v0123);

    // ror16
    v15121314 = _mm_ror_epi32(v15121314, 16);

    // add
    v101189 = _mm_add_epi32(v101189, v15121314);

    // xor
    v5674 = _mm_xor_si128(v5674, v101189);

    // ror12
    v5674 = _mm_ror_epi32(v5674, 12);

    // add
    v0123 = _mm_add_epi32(v0123, ms9111315);

    // add
    v0123 = _mm_add_epi32(v0123, v5674);

    // xor
    v15121314 = _mm_xor_si128(v15121314, v0123);

    // ror8
    v15121314 = _mm_ror_epi32(v15121314, 8);

    // add
    v101189 = _mm_add_epi32(v101189, v15121314);

    // xor
    v5674 = _mm_xor_si128(v5674, v101189);

    // ror7
    v5674 = _mm_ror_epi32(v5674, 7);

    // Store results back

    _mm_storeu_si128(v[..4].as_mut_ptr() as *mut _, v0123);

    let v4567 = _mm_shuffle_epi32(v5674, 0b0010_0001_0000_0011);
    _mm_storeu_si128(v[4..8].as_mut_ptr() as *mut _, v4567);

    let v891011 = _mm_shuffle_epi32(v101189, 0b0001_0000_0011_0010);
    _mm_storeu_si128(v[8..12].as_mut_ptr() as *mut _, v891011);

    let v12131415 = _mm_shuffle_epi32(v15121314, 0b0000_0011_0010_0001);
    _mm_storeu_si128(v[12..].as_mut_ptr() as *mut _, v12131415);
}

// NOTE: Writing out the whole round explicitly in this way gives better
// performance than we get if we factor out the G function. Perhaps the
// compiler doesn't notice that it can group all the adds together like we do
// here, even when G is inlined.
#[inline(always)]
unsafe fn blake2s_round_16x(v: &mut [__m512i; 16], m: &[__m512i; 16], r: usize) {
    // Select the message schedule based on the round.
    let s = SIGMA[r];

    // Mix the columns.
    g!(v, 0, 4, 8, 12, m[s[0] as usize], m[s[1] as usize]);
    g!(v, 1, 5, 9, 13, m[s[2] as usize], m[s[3] as usize]);
    g!(v, 2, 6, 10, 14, m[s[4] as usize], m[s[5] as usize]);
    g!(v, 3, 7, 11, 15, m[s[6] as usize], m[s[7] as usize]);

    // Mix the rows.
    g!(v, 0, 5, 10, 15, m[s[8] as usize], m[s[9] as usize]);
    g!(v, 1, 6, 11, 12, m[s[10] as usize], m[s[11] as usize]);
    g!(v, 2, 7, 8, 13, m[s[12] as usize], m[s[13] as usize]);
    g!(v, 3, 4, 9, 14, m[s[14] as usize], m[s[15] as usize]);
}

// #[target_feature(enable = "avx512f")]
// pub unsafe fn compress16(
//     h0: &mut StateWords,
//     h1: &mut StateWords,
//     h2: &mut StateWords,
//     h3: &mut StateWords,
//     h4: &mut StateWords,
//     h5: &mut StateWords,
//     h6: &mut StateWords,
//     h7: &mut StateWords,
//     h8: &mut StateWords,
//     h9: &mut StateWords,
//     h10: &mut StateWords,
//     h11: &mut StateWords,
//     h12: &mut StateWords,
//     h13: &mut StateWords,
//     h14: &mut StateWords,
//     h15: &mut StateWords,
//     msg0: &Block,
//     msg1: &Block,
//     msg2: &Block,
//     msg3: &Block,
//     msg4: &Block,
//     msg5: &Block,
//     msg6: &Block,
//     msg7: &Block,
//     msg8: &Block,
//     msg9: &Block,
//     msg10: &Block,
//     msg11: &Block,
//     msg12: &Block,
//     msg13: &Block,
//     msg14: &Block,
//     msg15: &Block,
//     count0: u64,
//     count1: u64,
//     count2: u64,
//     count3: u64,
//     count4: u64,
//     count5: u64,
//     count6: u64,
//     count7: u64,
//     count8: u64,
//     count9: u64,
//     count10: u64,
//     count11: u64,
//     count12: u64,
//     count13: u64,
//     count14: u64,
//     count15: u64,
//     lastblock0: u32,
//     lastblock1: u32,
//     lastblock2: u32,
//     lastblock3: u32,
//     lastblock4: u32,
//     lastblock5: u32,
//     lastblock6: u32,
//     lastblock7: u32,
//     lastblock8: u32,
//     lastblock9: u32,
//     lastblock10: u32,
//     lastblock11: u32,
//     lastblock12: u32,
//     lastblock13: u32,
//     lastblock14: u32,
//     lastblock15: u32,
//     lastnode0: u32,
//     lastnode1: u32,
//     lastnode2: u32,
//     lastnode3: u32,
//     lastnode4: u32,
//     lastnode5: u32,
//     lastnode6: u32,
//     lastnode7: u32,
//     lastnode8: u32,
//     lastnode9: u32,
//     lastnode10: u32,
//     lastnode11: u32,
//     lastnode12: u32,
//     lastnode13: u32,
//     lastnode14: u32,
//     lastnode15: u32,
// ) {
//     let mut h_vecs = [
//         loadu(h0.as_ptr()),
//         loadu(h1.as_ptr()),
//         loadu(h2.as_ptr()),
//         loadu(h3.as_ptr()),
//         loadu(h4.as_ptr()),
//         loadu(h5.as_ptr()),
//         loadu(h6.as_ptr()),
//         loadu(h7.as_ptr()),
//         loadu(h8.as_ptr()),
//         loadu(h9.as_ptr()),
//         loadu(h10.as_ptr()),
//         loadu(h11.as_ptr()),
//         loadu(h12.as_ptr()),
//         loadu(h13.as_ptr()),
//         loadu(h14.as_ptr()),
//         loadu(h15.as_ptr()),
//     ];
//     transpose_vecs(&mut h_vecs);

//     let count_low = load_512_from_16xu32(
//         count0 as u32,
//         count1 as u32,
//         count2 as u32,
//         count3 as u32,
//         count4 as u32,
//         count5 as u32,
//         count6 as u32,
//         count7 as u32,
//         count8 as u32,
//         count9 as u32,
//         count10 as u32,
//         count11 as u32,
//         count12 as u32,
//         count13 as u32,
//         count14 as u32,
//         count15 as u32,
//     );
//     let count_high = load_512_from_16xu32(
//         (count0 >> 32) as u32,
//         (count1 >> 32) as u32,
//         (count2 >> 32) as u32,
//         (count3 >> 32) as u32,
//         (count4 >> 32) as u32,
//         (count5 >> 32) as u32,
//         (count6 >> 32) as u32,
//         (count7 >> 32) as u32,
//         (count8 >> 32) as u32,
//         (count9 >> 32) as u32,
//         (count10 >> 32) as u32,
//         (count11 >> 32) as u32,
//         (count12 >> 32) as u32,
//         (count13 >> 32) as u32,
//         (count14 >> 32) as u32,
//         (count15 >> 32) as u32,
//     );
//     let lastblock = load_512_from_16xu32(
//         lastblock0 as u32,
//         lastblock1 as u32,
//         lastblock2 as u32,
//         lastblock3 as u32,
//         lastblock4 as u32,
//         lastblock5 as u32,
//         lastblock6 as u32,
//         lastblock7 as u32,
//         lastblock8 as u32,
//         lastblock9 as u32,
//         lastblock10 as u32,
//         lastblock11 as u32,
//         lastblock12 as u32,
//         lastblock13 as u32,
//         lastblock14 as u32,
//         lastblock15 as u32,
//     );
//     let lastnode = load_512_from_16xu32(
//         lastnode0 as u32,
//         lastnode1 as u32,
//         lastnode2 as u32,
//         lastnode3 as u32,
//         lastnode4 as u32,
//         lastnode5 as u32,
//         lastnode6 as u32,
//         lastnode7 as u32,
//         lastnode8 as u32,
//         lastnode9 as u32,
//         lastnode10 as u32,
//         lastnode11 as u32,
//         lastnode12 as u32,
//         lastnode13 as u32,
//         lastnode14 as u32,
//         lastnode15 as u32,
//     );

//     let msg_vecs = load_msg_vecs_interleave(
//         msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, msg8, msg9, msg10, msg11, msg12, msg13,
//         msg14, msg15,
//     );

//     compress16_transposed_inline(
//         &mut h_vecs,
//         &msg_vecs,
//         count_low,
//         count_high,
//         lastblock,
//         lastnode,
//     );

//     transpose_vecs(&mut h_vecs);

//     *h0 = mem::transmute(h_vecs[0]);
//     *h1 = mem::transmute(h_vecs[1]);
//     *h2 = mem::transmute(h_vecs[2]);
//     *h3 = mem::transmute(h_vecs[3]);
//     *h4 = mem::transmute(h_vecs[4]);
//     *h5 = mem::transmute(h_vecs[5]);
//     *h6 = mem::transmute(h_vecs[6]);
//     *h7 = mem::transmute(h_vecs[7]);
//     *h8 = mem::transmute(h_vecs[8]);
//     *h9 = mem::transmute(h_vecs[9]);
//     *h10 = mem::transmute(h_vecs[10]);
//     *h11 = mem::transmute(h_vecs[11]);
//     *h12 = mem::transmute(h_vecs[12]);
//     *h13 = mem::transmute(h_vecs[13]);
//     *h14 = mem::transmute(h_vecs[14]);
//     *h15 = mem::transmute(h_vecs[15]);
// }

// #[inline(always)]
// unsafe fn interleave128(a: __m512i, b: __m512i) -> (__m512i, __m512i) {
//     // (
//     //     // [low_b, low_a]
//     //     _mm256_permute2x128_si256(a, b, 0b0010_0000),
//     //     // [high_b, high_a]
//     //     _mm256_permute2x128_si256(a, b, 0b0011_0001),
//     // )

//     // TODO: correct permutations
//     (a, b)
// }

// #[cfg(test)]
// fn cast_out(x: __m512i) -> [u32; 16] {
//     unsafe { mem::transmute(x) }
// }

// #[cfg(test)]
// #[test]
// fn test_interleave128() {
//     #[target_feature(enable = "avx512f")]
//     unsafe fn inner() {
//         let a = load_512_from_16xu32(10, 11, 12, 13, 14, 15, 16, 17);
//         let b = load_512_from_16xu32(20, 21, 22, 23, 24, 25, 26, 27);

//         let expected_a = load_512_from_16xu32(10, 11, 12, 13, 20, 21, 22, 23);
//         let expected_b = load_512_from_16xu32(14, 15, 16, 17, 24, 25, 26, 27);

//         let (out_a, out_b) = interleave128(a, b);

//         assert_eq!(cast_out(expected_a), cast_out(out_a));
//         assert_eq!(cast_out(expected_b), cast_out(out_b));
//     }

//     #[cfg(feature = "std")]
//     {
//         if is_x86_feature_detected!("avx512f") {
//             unsafe {
//                 inner();
//             }
//         }
//     }
// }

// #[inline(always)]
// unsafe fn load_2x512(msg: &[u8; BLOCKBYTES]) -> (__m512i, __m512i) {
//     (
//         _mm512_loadu_si512(msg.as_ptr() as *const __m512i),
//         _mm512_loadu_si512((msg.as_ptr() as *const __m512i).add(1)),
//     )
// }

// #[cfg(test)]
// #[test]
// fn test_load_2x512() {
//     #[target_feature(enable = "avx512f")]
//     unsafe fn inner() {
//         let input: [u64; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
//         let input_bytes: [u8; BLOCKBYTES] = mem::transmute(input);
//         let (out_a, out_b) = load_2x512(&input_bytes);

//         let expected_a = load_512_from_16xu32(0, 0, 1, 0, 2, 0, 3, 0);
//         let expected_b = load_512_from_16xu32(4, 0, 5, 0, 6, 0, 7, 0);

//         assert_eq!(cast_out(expected_a), cast_out(out_a));
//         assert_eq!(cast_out(expected_b), cast_out(out_b));
//     }

//     #[cfg(feature = "std")]
//     {
//         if is_x86_feature_detected!("avx512f") {
//             unsafe {
//                 inner();
//             }
//         }
//     }
// }

// #[inline(always)]
// unsafe fn transpose_vecs(vecs: &mut [__m512i; 8]) {
//     // Interleave 32-bit lanes. The low unpack is lanes 00/11/44/55, and the high is 22/33/66/77.
//     let ab_0145 = _mm512_unpacklo_epi32(vecs[0], vecs[1]);
//     let ab_2367 = _mm512_unpackhi_epi32(vecs[0], vecs[1]);
//     let cd_0145 = _mm512_unpacklo_epi32(vecs[2], vecs[3]);
//     let cd_2367 = _mm512_unpackhi_epi32(vecs[2], vecs[3]);
//     let ef_0145 = _mm512_unpacklo_epi32(vecs[4], vecs[5]);
//     let ef_2367 = _mm512_unpackhi_epi32(vecs[4], vecs[5]);
//     let gh_0145 = _mm512_unpacklo_epi32(vecs[6], vecs[7]);
//     let gh_2367 = _mm512_unpackhi_epi32(vecs[6], vecs[7]);

//     let ij_0145 = _mm512_unpacklo_epi32(vecs[8], vecs[9]);
//     let ij_2367 = _mm512_unpackhi_epi32(vecs[8], vecs[9]);
//     let kl_0145 = _mm512_unpacklo_epi32(vecs[10], vecs[11]);
//     let kl_2367 = _mm512_unpackhi_epi32(vecs[10], vecs[11]);
//     let mn_0145 = _mm512_unpacklo_epi32(vecs[12], vecs[13]);
//     let mn_2367 = _mm512_unpackhi_epi32(vecs[12], vecs[13]);
//     let op_0145 = _mm512_unpacklo_epi32(vecs[14], vecs[15]);
//     let op_2367 = _mm512_unpackhi_epi32(vecs[14], vecs[15]);

//     // Interleave 64-bit lanes. The low unpack is lanes 00/22 and the high is 11/33.
//     let abcd_04 = _mm512_unpacklo_epi64(ab_0145, cd_0145);
//     let abcd_15 = _mm512_unpackhi_epi64(ab_0145, cd_0145);
//     let abcd_26 = _mm512_unpacklo_epi64(ab_2367, cd_2367);
//     let abcd_37 = _mm512_unpackhi_epi64(ab_2367, cd_2367);
//     let efgh_04 = _mm512_unpacklo_epi64(ef_0145, gh_0145);
//     let efgh_15 = _mm512_unpackhi_epi64(ef_0145, gh_0145);
//     let efgh_26 = _mm512_unpacklo_epi64(ef_2367, gh_2367);
//     let efgh_37 = _mm512_unpackhi_epi64(ef_2367, gh_2367);

//     let ijkl_04 = _mm512_unpacklo_epi64(ij_0145, kl_0145);
//     let ijkl_15 = _mm512_unpackhi_epi64(ij_0145, kl_0145);
//     let ijkl_26 = _mm512_unpacklo_epi64(ij_2367, kl_2367);
//     let ijkl_37 = _mm512_unpackhi_epi64(ij_2367, kl_2367);
//     let mnop_04 = _mm512_unpacklo_epi64(mn_0145, op_0145);
//     let mnop_15 = _mm512_unpackhi_epi64(mn_0145, op_0145);
//     let mnop_26 = _mm512_unpacklo_epi64(mn_2367, op_2367);
//     let mnop_37 = _mm512_unpackhi_epi64(mn_2367, op_2367);

//     // Interleave 128-bit lanes.
//     let (abcdefg_0, abcdefg_4) = interleave128(abcd_04, efgh_04);
//     let (abcdefg_1, abcdefg_5) = interleave128(abcd_15, efgh_15);
//     let (abcdefg_2, abcdefg_6) = interleave128(abcd_26, efgh_26);
//     let (abcdefg_3, abcdefg_7) = interleave128(abcd_37, efgh_37);

//     let (ijklmnop_0, ijklmnop_4) = interleave128(ijkl_04, mnop_04);
//     let (ijklmnop_1, ijklmnop_5) = interleave128(ijkl_15, mnop_15);
//     let (ijklmnop_2, ijklmnop_6) = interleave128(ijkl_26, mnop_26);
//     let (ijklmnop_3, ijklmnop_7) = interleave128(ijkl_37, mnop_37);

//     vecs[0] = abcdefg_0;
//     vecs[1] = abcdefg_1;
//     vecs[2] = abcdefg_2;
//     vecs[3] = abcdefg_3;
//     vecs[4] = abcdefg_4;
//     vecs[5] = abcdefg_5;
//     vecs[6] = abcdefg_6;
//     vecs[7] = abcdefg_7;
//     vecs[8] = ijklmnop_0;
//     vecs[9] = ijklmnop_1;
//     vecs[10] = ijklmnop_2;
//     vecs[11] = ijklmnop_3;
//     vecs[12] = ijklmnop_4;
//     vecs[13] = ijklmnop_5;
//     vecs[14] = ijklmnop_6;
//     vecs[15] = ijklmnop_7;
// }

// #[target_feature(enable = "avx512f")]
// pub unsafe fn vectorize_words16(words: &mut [AlignedWords8; 16]) {
//     let vecs = &mut *(words as *mut _ as *mut [__m512i; 8]);
//     transpose_vecs(vecs);
// }

// #[cfg(test)]
// #[test]
// fn test_transpose_vecs() {
//     #[target_feature(enable = "avx512f")]
//     unsafe fn inner() {
//         let vec_a = load_512_from_16xu32(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
//         let vec_b = load_512_from_16xu32(0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17);
//         let vec_c = load_512_from_16xu32(0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27);
//         let vec_d = load_512_from_16xu32(0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37);
//         let vec_e = load_512_from_16xu32(0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47);
//         let vec_f = load_512_from_16xu32(0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57);
//         let vec_g = load_512_from_16xu32(0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67);
//         let vec_h = load_512_from_16xu32(0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77);

//         let expected_a = load_512_from_16xu32(0x00, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70);
//         let expected_b = load_512_from_16xu32(0x01, 0x11, 0x21, 0x31, 0x41, 0x51, 0x61, 0x71);
//         let expected_c = load_512_from_16xu32(0x02, 0x12, 0x22, 0x32, 0x42, 0x52, 0x62, 0x72);
//         let expected_d = load_512_from_16xu32(0x03, 0x13, 0x23, 0x33, 0x43, 0x53, 0x63, 0x73);
//         let expected_e = load_512_from_16xu32(0x04, 0x14, 0x24, 0x34, 0x44, 0x54, 0x64, 0x74);
//         let expected_f = load_512_from_16xu32(0x05, 0x15, 0x25, 0x35, 0x45, 0x55, 0x65, 0x75);
//         let expected_g = load_512_from_16xu32(0x06, 0x16, 0x26, 0x36, 0x46, 0x56, 0x66, 0x76);
//         let expected_h = load_512_from_16xu32(0x07, 0x17, 0x27, 0x37, 0x47, 0x57, 0x67, 0x77);

//         let mut interleaved = [vec_a, vec_b, vec_c, vec_d, vec_e, vec_f, vec_g, vec_h];
//         transpose_vecs(&mut interleaved);

//         let [out_a, out_b, out_c, out_d, out_e, out_f, out_g, out_h] = interleaved;
//         assert_eq!(cast_out(expected_a), cast_out(out_a));
//         assert_eq!(cast_out(expected_b), cast_out(out_b));
//         assert_eq!(cast_out(expected_c), cast_out(out_c));
//         assert_eq!(cast_out(expected_d), cast_out(out_d));
//         assert_eq!(cast_out(expected_e), cast_out(out_e));
//         assert_eq!(cast_out(expected_f), cast_out(out_f));
//         assert_eq!(cast_out(expected_g), cast_out(out_g));
//         assert_eq!(cast_out(expected_h), cast_out(out_h));

//         // Check that interleaving again undoes the operation.
//         let mut deinterleaved = [out_a, out_b, out_c, out_d, out_e, out_f, out_g, out_h];
//         transpose_vecs(&mut deinterleaved);
//         let [out2_a, out2_b, out2_c, out2_d, out2_e, out2_f, out2_g, out2_h] = deinterleaved;
//         assert_eq!(cast_out(vec_a), cast_out(out2_a));
//         assert_eq!(cast_out(vec_b), cast_out(out2_b));
//         assert_eq!(cast_out(vec_c), cast_out(out2_c));
//         assert_eq!(cast_out(vec_d), cast_out(out2_d));
//         assert_eq!(cast_out(vec_e), cast_out(out2_e));
//         assert_eq!(cast_out(vec_f), cast_out(out2_f));
//         assert_eq!(cast_out(vec_g), cast_out(out2_g));
//         assert_eq!(cast_out(vec_h), cast_out(out2_h));
//     }

//     #[cfg(feature = "std")]
//     {
//         if is_x86_feature_detected!("avx512f") {
//             unsafe {
//                 inner();
//             }
//         }
//     }
// }

// #[inline(always)]
// unsafe fn load_msg_vecs_interleave(
//     msg_a: &[u8; BLOCKBYTES],
//     msg_b: &[u8; BLOCKBYTES],
//     msg_c: &[u8; BLOCKBYTES],
//     msg_d: &[u8; BLOCKBYTES],
//     msg_e: &[u8; BLOCKBYTES],
//     msg_f: &[u8; BLOCKBYTES],
//     msg_g: &[u8; BLOCKBYTES],
//     msg_h: &[u8; BLOCKBYTES],
//     msg_i: &[u8; BLOCKBYTES],
//     msg_j: &[u8; BLOCKBYTES],
//     msg_k: &[u8; BLOCKBYTES],
//     msg_l: &[u8; BLOCKBYTES],
//     msg_m: &[u8; BLOCKBYTES],
//     msg_n: &[u8; BLOCKBYTES],
//     msg_o: &[u8; BLOCKBYTES],
//     msg_p: &[u8; BLOCKBYTES],
// ) -> [__m512i; 16] {
//     let (front_a, back_a) = load_2x512(msg_a);
//     let (front_b, back_b) = load_2x512(msg_b);
//     let (front_c, back_c) = load_2x512(msg_c);
//     let (front_d, back_d) = load_2x512(msg_d);
//     let (front_e, back_e) = load_2x512(msg_e);
//     let (front_f, back_f) = load_2x512(msg_f);
//     let (front_g, back_g) = load_2x512(msg_g);
//     let (front_h, back_h) = load_2x512(msg_h);
//     let (front_i, back_i) = load_2x512(msg_i);
//     let (front_j, back_j) = load_2x512(msg_j);
//     let (front_k, back_k) = load_2x512(msg_k);
//     let (front_l, back_l) = load_2x512(msg_l);
//     let (front_m, back_m) = load_2x512(msg_m);
//     let (front_n, back_n) = load_2x512(msg_n);
//     let (front_o, back_o) = load_2x512(msg_o);
//     let (front_p, back_p) = load_2x512(msg_p);

//     let mut front_interleaved = [
//         front_a, front_b, front_c, front_d, front_e, front_f, front_g, front_h, front_i, front_j,
//         front_k, front_l, front_m, front_n, front_o, front_p,
//     ];
//     transpose_vecs(&mut front_interleaved);
//     let mut back_interleaved = [
//         back_a, back_b, back_c, back_d, back_e, back_f, back_g, back_h, back_i, back_j, back_k,
//         back_l, back_m, back_n, back_o, back_p,
//     ];
//     transpose_vecs(&mut back_interleaved);

//     [
//         front_interleaved[0],
//         front_interleaved[1],
//         front_interleaved[2],
//         front_interleaved[3],
//         front_interleaved[4],
//         front_interleaved[5],
//         front_interleaved[6],
//         front_interleaved[7],
//         front_interleaved[8],
//         front_interleaved[9],
//         front_interleaved[10],
//         front_interleaved[11],
//         front_interleaved[12],
//         front_interleaved[13],
//         front_interleaved[14],
//         front_interleaved[15],
//         back_interleaved[0],
//         back_interleaved[1],
//         back_interleaved[2],
//         back_interleaved[3],
//         back_interleaved[4],
//         back_interleaved[5],
//         back_interleaved[6],
//         back_interleaved[7],
//         back_interleaved[8],
//         back_interleaved[9],
//         back_interleaved[10],
//         back_interleaved[11],
//         back_interleaved[12],
//         back_interleaved[13],
//         back_interleaved[14],
//         back_interleaved[15],
//     ]
// }

// // This function assumes that the state is in transposed form, but not
// // necessarily aligned. It accepts input in the usual form of contiguous bytes,
// // and it pays the cost of transposing the input.
// #[target_feature(enable = "avx512f")]
// pub unsafe fn compress16_vectorized(
//     states: &mut [AlignedWords8; 16],
//     msg0: &Block,
//     msg1: &Block,
//     msg2: &Block,
//     msg3: &Block,
//     msg4: &Block,
//     msg5: &Block,
//     msg6: &Block,
//     msg7: &Block,
//     msg8: &Block,
//     msg9: &Block,
//     msg10: &Block,
//     msg11: &Block,
//     msg12: &Block,
//     msg13: &Block,
//     msg14: &Block,
//     msg15: &Block,
//     count_low: &AlignedWords8,
//     count_high: &AlignedWords8,
//     lastblock: &AlignedWords8,
//     lastnode: &AlignedWords8,
// ) {
//     let mut h_vecs = &mut *(states as *mut _ as *mut [__m512i; 8]);

//     let msg_vecs = load_msg_vecs_interleave(
//         msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, msg8, msg9, msg10, msg11, msg12, msg13,
//         msg14, msg15,
//     );

//     compress16_transposed_inline(
//         &mut h_vecs,
//         &msg_vecs,
//         mem::transmute(*count_low),
//         mem::transmute(*count_high),
//         mem::transmute(*lastblock),
//         mem::transmute(*lastnode),
//     );
// }

#[target_feature(enable = "avx512f")]
pub unsafe fn compress16_transposed_all(
    h_vecs: &mut [__m512i; 8],
    msg_vecs: &[__m512i; 16],
    count_low: __m512i,
    count_high: __m512i,
    lastblock: __m512i,
    lastnode: __m512i,
) {
    compress16_transposed_inline(h_vecs, msg_vecs, count_low, count_high, lastblock, lastnode);
}

// This core function assumes that both the state words and the message blocks
// have been transposed across vectors. So the first state vector contains the
// first word of each of the 8 states, and the first message vector contains
// the first word of each of the 8 message blocks. Defining the core this way
// allows us to keep either the state or the message in transposed form in some
// cases, to avoid paying the cost of transposing them.
#[inline(always)]
unsafe fn compress16_transposed_inline(
    h_vecs: &mut [__m512i; 8],
    msg_vecs: &[__m512i; 16],
    count_low: __m512i,
    count_high: __m512i,
    lastblock: __m512i,
    lastnode: __m512i,
) {
    let mut v = [
        h_vecs[0],
        h_vecs[1],
        h_vecs[2],
        h_vecs[3],
        h_vecs[4],
        h_vecs[5],
        h_vecs[6],
        h_vecs[7],
        load_512_from_u32(IV[0]),
        load_512_from_u32(IV[1]),
        load_512_from_u32(IV[2]),
        load_512_from_u32(IV[3]),
        xor(load_512_from_u32(IV[4]), count_low),
        xor(load_512_from_u32(IV[5]), count_high),
        xor(load_512_from_u32(IV[6]), lastblock),
        xor(load_512_from_u32(IV[7]), lastnode),
    ];

    blake2s_round_16x(&mut v, &msg_vecs, 0);
    blake2s_round_16x(&mut v, &msg_vecs, 1);
    blake2s_round_16x(&mut v, &msg_vecs, 2);
    blake2s_round_16x(&mut v, &msg_vecs, 3);
    blake2s_round_16x(&mut v, &msg_vecs, 4);
    blake2s_round_16x(&mut v, &msg_vecs, 5);
    blake2s_round_16x(&mut v, &msg_vecs, 6);
    blake2s_round_16x(&mut v, &msg_vecs, 7);
    blake2s_round_16x(&mut v, &msg_vecs, 8);
    blake2s_round_16x(&mut v, &msg_vecs, 9);

    h_vecs[0] = xor(h_vecs[0], xor(v[0], v[8]));
    h_vecs[1] = xor(h_vecs[1], xor(v[1], v[9]));
    h_vecs[2] = xor(h_vecs[2], xor(v[2], v[10]));
    h_vecs[3] = xor(h_vecs[3], xor(v[3], v[11]));
    h_vecs[4] = xor(h_vecs[4], xor(v[4], v[12]));
    h_vecs[5] = xor(h_vecs[5], xor(v[5], v[13]));
    h_vecs[6] = xor(h_vecs[6], xor(v[6], v[14]));
    h_vecs[7] = xor(h_vecs[7], xor(v[7], v[15]));
}

// #[inline(always)]
// unsafe fn export_hashes(h_vecs: &[__m512i; 8], hash_length: u8) -> [Hash; 16] {
//     // Interleave is its own inverse.
//     let mut deinterleaved = *h_vecs;
//     transpose_vecs(&mut deinterleaved);
//     // BLAKE2 and x86 both use little-endian representation, so we can just transmute the word
//     // bytes out of each de-interleaved vector.
//     [
//         Hash {
//             len: hash_length,
//             bytes: mem::transmute(deinterleaved[0]),
//         },
//         Hash {
//             len: hash_length,
//             bytes: mem::transmute(deinterleaved[1]),
//         },
//         Hash {
//             len: hash_length,
//             bytes: mem::transmute(deinterleaved[2]),
//         },
//         Hash {
//             len: hash_length,
//             bytes: mem::transmute(deinterleaved[3]),
//         },
//         Hash {
//             len: hash_length,
//             bytes: mem::transmute(deinterleaved[4]),
//         },
//         Hash {
//             len: hash_length,
//             bytes: mem::transmute(deinterleaved[5]),
//         },
//         Hash {
//             len: hash_length,
//             bytes: mem::transmute(deinterleaved[6]),
//         },
//         Hash {
//             len: hash_length,
//             bytes: mem::transmute(deinterleaved[7]),
//         },
//         Hash {
//             len: hash_length,
//             bytes: mem::transmute(deinterleaved[8]),
//         },
//         Hash {
//             len: hash_length,
//             bytes: mem::transmute(deinterleaved[9]),
//         },
//         Hash {
//             len: hash_length,
//             bytes: mem::transmute(deinterleaved[10]),
//         },
//         Hash {
//             len: hash_length,
//             bytes: mem::transmute(deinterleaved[3]),
//         },
//         Hash {
//             len: hash_length,
//             bytes: mem::transmute(deinterleaved[4]),
//         },
//         Hash {
//             len: hash_length,
//             bytes: mem::transmute(deinterleaved[5]),
//         },
//         Hash {
//             len: hash_length,
//             bytes: mem::transmute(deinterleaved[6]),
//         },
//         Hash {
//             len: hash_length,
//             bytes: mem::transmute(deinterleaved[7]),
//         },
//     ]
// }

// #[target_feature(enable = "avx512f")]
// pub unsafe fn hash16_exact(
//     // TODO: Separate params for each input.
//     params: &Params,
//     input0: &[u8],
//     input1: &[u8],
//     input2: &[u8],
//     input3: &[u8],
//     input4: &[u8],
//     input5: &[u8],
//     input6: &[u8],
//     input7: &[u8],
//     input8: &[u8],
//     input9: &[u8],
//     input10: &[u8],
//     input11: &[u8],
//     input12: &[u8],
//     input13: &[u8],
//     input14: &[u8],
//     input15: &[u8],
// ) -> [Hash; 16] {
//     // INVARIANTS! The caller must assert:
//     //   1. The inputs are the same length.
//     //   2. The inputs are a multiple of the block size.
//     //   3. The inputs aren't empty.

//     let param_words = params.make_words();
//     // This creates word vectors in an aready-transposed position.
//     let mut h_vecs = [
//         load_512_from_u32(param_words[0]),
//         load_512_from_u32(param_words[1]),
//         load_512_from_u32(param_words[2]),
//         load_512_from_u32(param_words[3]),
//         load_512_from_u32(param_words[4]),
//         load_512_from_u32(param_words[5]),
//         load_512_from_u32(param_words[6]),
//         load_512_from_u32(param_words[7]),
//         load_512_from_u32(param_words[8]),
//         load_512_from_u32(param_words[9]),
//         load_512_from_u32(param_words[10]),
//         load_512_from_u32(param_words[11]),
//         load_512_from_u32(param_words[12]),
//         load_512_from_u32(param_words[13]),
//         load_512_from_u32(param_words[14]),
//         load_512_from_u32(param_words[15]),
//     ];
//     let len = input0.len();
//     let mut count = 0;

//     loop {
//         // Use pointer casts to avoid bounds checks here. The caller has to assert that these exact
//         // bounds are valid. Note that if these bounds were wrong, we'd get the wrong hash in any
//         // case, because count is an input to the compression function.
//         let msg0 = &*(input0.as_ptr().add(count) as *const Block);
//         let msg1 = &*(input1.as_ptr().add(count) as *const Block);
//         let msg2 = &*(input2.as_ptr().add(count) as *const Block);
//         let msg3 = &*(input3.as_ptr().add(count) as *const Block);
//         let msg4 = &*(input4.as_ptr().add(count) as *const Block);
//         let msg5 = &*(input5.as_ptr().add(count) as *const Block);
//         let msg6 = &*(input6.as_ptr().add(count) as *const Block);
//         let msg7 = &*(input7.as_ptr().add(count) as *const Block);
//         let msg8 = &*(input8.as_ptr().add(count) as *const Block);
//         let msg9 = &*(input9.as_ptr().add(count) as *const Block);
//         let msg10 = &*(inpu102.as_ptr().add(count) as *const Block);
//         let msg11 = &*(inpu113.as_ptr().add(count) as *const Block);
//         let msg12 = &*(inpu124.as_ptr().add(count) as *const Block);
//         let msg13 = &*(inpu135.as_ptr().add(count) as *const Block);
//         let msg14 = &*(inpu146.as_ptr().add(count) as *const Block);
//         let msg15 = &*(inpu157.as_ptr().add(count) as *const Block);
//         count += BLOCKBYTES;
//         let count_low = load_512_from_u32(count as u32);
//         let count_high = load_512_from_u32((count as u64 >> 32) as u32);
//         let lastblock = load_512_from_u32(if count == len { !0 } else { 0 });
//         let lastnode = load_512_from_u32(if params.last_node && count == len {
//             !0
//         } else {
//             0
//         });
//         let msg_vecs = load_msg_vecs_interleave(
//             msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, msg8, msg9, msg10, msg11, msg12, msg13,
//             msg14, msg15,
//         );
//         compress16_transposed_inline(
//             &mut h_vecs,
//             &msg_vecs,
//             count_low,
//             count_high,
//             lastblock,
//             lastnode,
//         );
//         if count == len {
//             return export_hashes(&h_vecs, params.hash_length);
//         }
//     }
// }
