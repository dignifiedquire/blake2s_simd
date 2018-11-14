#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use byteorder::{ByteOrder, LittleEndian};
use core::mem;

use crate::Aligned8x8Words;
use crate::Block;
use crate::Hash;
use crate::Params;
use crate::StateWords;
use crate::BLOCKBYTES;
use crate::IV;
use crate::SIGMA;

#[inline(always)]
unsafe fn add(a: __m256i, b: __m256i) -> __m256i {
    _mm256_add_epi32(a, b)
}

#[inline(always)]
unsafe fn xor(a: __m256i, b: __m256i) -> __m256i {
    _mm256_xor_si256(a, b)
}

#[inline(always)]
unsafe fn rot16(x: __m256i) -> __m256i {
    _mm256_shuffle_epi8(
        x,
        _mm256_set_epi8(
            13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2, 13, 12, 15, 14, 9, 8, 11, 10, 5,
            4, 7, 6, 1, 0, 3, 2,
        ),
    )
}

#[inline(always)]
unsafe fn rot12(x: __m256i) -> __m256i {
    _mm256_or_si256(_mm256_srli_epi32(x, 12), _mm256_slli_epi32(x, 20))
}

#[inline(always)]
unsafe fn rot8(x: __m256i) -> __m256i {
    _mm256_shuffle_epi8(
        x,
        _mm256_set_epi8(
            12, 15, 14, 13, 8, 11, 10, 9, 4, 7, 6, 5, 0, 3, 2, 1, 12, 15, 14, 13, 8, 11, 10, 9, 4,
            7, 6, 5, 0, 3, 2, 1,
        ),
    )
}

#[inline(always)]
unsafe fn rot7(x: __m256i) -> __m256i {
    _mm256_or_si256(_mm256_srli_epi32(x, 7), _mm256_slli_epi32(x, 25))
}

#[inline(always)]
unsafe fn load_256_from_u32(x: u32) -> __m256i {
    _mm256_set1_epi32(x as i32)
}

#[inline(always)]
unsafe fn load_256_from_8xu32(
    x1: u32,
    x2: u32,
    x3: u32,
    x4: u32,
    x5: u32,
    x6: u32,
    x7: u32,
    x8: u32,
) -> __m256i {
    // NOTE: This order of arguments for _mm256_set_epi32 is the reverse of how the ints come out
    // when you transmute them back into an array of u32's.
    _mm256_set_epi32(
        x8 as i32, x7 as i32, x6 as i32, x5 as i32, x4 as i32, x3 as i32, x2 as i32, x1 as i32,
    )
}

#[inline(always)]
unsafe fn load_msg_vec(
    msg0: &Block,
    msg1: &Block,
    msg2: &Block,
    msg3: &Block,
    msg4: &Block,
    msg5: &Block,
    msg6: &Block,
    msg7: &Block,
    i: usize,
) -> __m256i {
    load_256_from_8xu32(
        LittleEndian::read_u32(&msg0[4 * i..]),
        LittleEndian::read_u32(&msg1[4 * i..]),
        LittleEndian::read_u32(&msg2[4 * i..]),
        LittleEndian::read_u32(&msg3[4 * i..]),
        LittleEndian::read_u32(&msg4[4 * i..]),
        LittleEndian::read_u32(&msg5[4 * i..]),
        LittleEndian::read_u32(&msg6[4 * i..]),
        LittleEndian::read_u32(&msg7[4 * i..]),
    )
}

#[inline(always)]
pub unsafe fn load_msg_vecs_naive(
    msg0: &Block,
    msg1: &Block,
    msg2: &Block,
    msg3: &Block,
    msg4: &Block,
    msg5: &Block,
    msg6: &Block,
    msg7: &Block,
) -> [__m256i; 16] {
    [
        load_msg_vec(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, 0),
        load_msg_vec(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, 1),
        load_msg_vec(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, 2),
        load_msg_vec(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, 3),
        load_msg_vec(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, 4),
        load_msg_vec(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, 5),
        load_msg_vec(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, 6),
        load_msg_vec(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, 7),
        load_msg_vec(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, 8),
        load_msg_vec(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, 9),
        load_msg_vec(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, 10),
        load_msg_vec(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, 11),
        load_msg_vec(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, 12),
        load_msg_vec(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, 13),
        load_msg_vec(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, 14),
        load_msg_vec(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, 15),
    ]
}

#[inline(always)]
unsafe fn blake2s_round_8x(v: &mut [__m256i; 16], m: &[__m256i; 16], r: usize) {
    v[0] = add(v[0], m[SIGMA[r][0] as usize]);
    v[1] = add(v[1], m[SIGMA[r][2] as usize]);
    v[2] = add(v[2], m[SIGMA[r][4] as usize]);
    v[3] = add(v[3], m[SIGMA[r][6] as usize]);
    v[0] = add(v[0], v[4]);
    v[1] = add(v[1], v[5]);
    v[2] = add(v[2], v[6]);
    v[3] = add(v[3], v[7]);
    v[12] = xor(v[12], v[0]);
    v[13] = xor(v[13], v[1]);
    v[14] = xor(v[14], v[2]);
    v[15] = xor(v[15], v[3]);
    v[12] = rot16(v[12]);
    v[13] = rot16(v[13]);
    v[14] = rot16(v[14]);
    v[15] = rot16(v[15]);
    v[8] = add(v[8], v[12]);
    v[9] = add(v[9], v[13]);
    v[10] = add(v[10], v[14]);
    v[11] = add(v[11], v[15]);
    v[4] = xor(v[4], v[8]);
    v[5] = xor(v[5], v[9]);
    v[6] = xor(v[6], v[10]);
    v[7] = xor(v[7], v[11]);
    v[4] = rot12(v[4]);
    v[5] = rot12(v[5]);
    v[6] = rot12(v[6]);
    v[7] = rot12(v[7]);
    v[0] = add(v[0], m[SIGMA[r][1] as usize]);
    v[1] = add(v[1], m[SIGMA[r][3] as usize]);
    v[2] = add(v[2], m[SIGMA[r][5] as usize]);
    v[3] = add(v[3], m[SIGMA[r][7] as usize]);
    v[0] = add(v[0], v[4]);
    v[1] = add(v[1], v[5]);
    v[2] = add(v[2], v[6]);
    v[3] = add(v[3], v[7]);
    v[12] = xor(v[12], v[0]);
    v[13] = xor(v[13], v[1]);
    v[14] = xor(v[14], v[2]);
    v[15] = xor(v[15], v[3]);
    v[12] = rot8(v[12]);
    v[13] = rot8(v[13]);
    v[14] = rot8(v[14]);
    v[15] = rot8(v[15]);
    v[8] = add(v[8], v[12]);
    v[9] = add(v[9], v[13]);
    v[10] = add(v[10], v[14]);
    v[11] = add(v[11], v[15]);
    v[4] = xor(v[4], v[8]);
    v[5] = xor(v[5], v[9]);
    v[6] = xor(v[6], v[10]);
    v[7] = xor(v[7], v[11]);
    v[4] = rot7(v[4]);
    v[5] = rot7(v[5]);
    v[6] = rot7(v[6]);
    v[7] = rot7(v[7]);

    v[0] = add(v[0], m[SIGMA[r][8] as usize]);
    v[1] = add(v[1], m[SIGMA[r][10] as usize]);
    v[2] = add(v[2], m[SIGMA[r][12] as usize]);
    v[3] = add(v[3], m[SIGMA[r][14] as usize]);
    v[0] = add(v[0], v[5]);
    v[1] = add(v[1], v[6]);
    v[2] = add(v[2], v[7]);
    v[3] = add(v[3], v[4]);
    v[15] = xor(v[15], v[0]);
    v[12] = xor(v[12], v[1]);
    v[13] = xor(v[13], v[2]);
    v[14] = xor(v[14], v[3]);
    v[15] = rot16(v[15]);
    v[12] = rot16(v[12]);
    v[13] = rot16(v[13]);
    v[14] = rot16(v[14]);
    v[10] = add(v[10], v[15]);
    v[11] = add(v[11], v[12]);
    v[8] = add(v[8], v[13]);
    v[9] = add(v[9], v[14]);
    v[5] = xor(v[5], v[10]);
    v[6] = xor(v[6], v[11]);
    v[7] = xor(v[7], v[8]);
    v[4] = xor(v[4], v[9]);
    v[5] = rot12(v[5]);
    v[6] = rot12(v[6]);
    v[7] = rot12(v[7]);
    v[4] = rot12(v[4]);
    v[0] = add(v[0], m[SIGMA[r][9] as usize]);
    v[1] = add(v[1], m[SIGMA[r][11] as usize]);
    v[2] = add(v[2], m[SIGMA[r][13] as usize]);
    v[3] = add(v[3], m[SIGMA[r][15] as usize]);
    v[0] = add(v[0], v[5]);
    v[1] = add(v[1], v[6]);
    v[2] = add(v[2], v[7]);
    v[3] = add(v[3], v[4]);
    v[15] = xor(v[15], v[0]);
    v[12] = xor(v[12], v[1]);
    v[13] = xor(v[13], v[2]);
    v[14] = xor(v[14], v[3]);
    v[15] = rot8(v[15]);
    v[12] = rot8(v[12]);
    v[13] = rot8(v[13]);
    v[14] = rot8(v[14]);
    v[10] = add(v[10], v[15]);
    v[11] = add(v[11], v[12]);
    v[8] = add(v[8], v[13]);
    v[9] = add(v[9], v[14]);
    v[5] = xor(v[5], v[10]);
    v[6] = xor(v[6], v[11]);
    v[7] = xor(v[7], v[8]);
    v[4] = xor(v[4], v[9]);
    v[5] = rot7(v[5]);
    v[6] = rot7(v[6]);
    v[7] = rot7(v[7]);
    v[4] = rot7(v[4]);
}

#[target_feature(enable = "avx2")]
pub unsafe fn compress8(
    h0: &mut StateWords,
    h1: &mut StateWords,
    h2: &mut StateWords,
    h3: &mut StateWords,
    h4: &mut StateWords,
    h5: &mut StateWords,
    h6: &mut StateWords,
    h7: &mut StateWords,
    msg0: &Block,
    msg1: &Block,
    msg2: &Block,
    msg3: &Block,
    msg4: &Block,
    msg5: &Block,
    msg6: &Block,
    msg7: &Block,
    count0: u64,
    count1: u64,
    count2: u64,
    count3: u64,
    count4: u64,
    count5: u64,
    count6: u64,
    count7: u64,
    lastblock0: u32,
    lastblock1: u32,
    lastblock2: u32,
    lastblock3: u32,
    lastblock4: u32,
    lastblock5: u32,
    lastblock6: u32,
    lastblock7: u32,
    lastnode0: u32,
    lastnode1: u32,
    lastnode2: u32,
    lastnode3: u32,
    lastnode4: u32,
    lastnode5: u32,
    lastnode6: u32,
    lastnode7: u32,
) {
    let mut h_vecs = interleave_vecs(
        _mm256_loadu_si256(h0 as *const StateWords as *const __m256i),
        _mm256_loadu_si256(h1 as *const StateWords as *const __m256i),
        _mm256_loadu_si256(h2 as *const StateWords as *const __m256i),
        _mm256_loadu_si256(h3 as *const StateWords as *const __m256i),
        _mm256_loadu_si256(h4 as *const StateWords as *const __m256i),
        _mm256_loadu_si256(h5 as *const StateWords as *const __m256i),
        _mm256_loadu_si256(h6 as *const StateWords as *const __m256i),
        _mm256_loadu_si256(h7 as *const StateWords as *const __m256i),
    );
    let count_low = load_256_from_8xu32(
        count0 as u32,
        count1 as u32,
        count2 as u32,
        count3 as u32,
        count4 as u32,
        count5 as u32,
        count6 as u32,
        count7 as u32,
    );
    let count_high = load_256_from_8xu32(
        (count0 >> 32) as u32,
        (count1 >> 32) as u32,
        (count2 >> 32) as u32,
        (count3 >> 32) as u32,
        (count4 >> 32) as u32,
        (count5 >> 32) as u32,
        (count6 >> 32) as u32,
        (count7 >> 32) as u32,
    );
    let lastblock = load_256_from_8xu32(
        lastblock0 as u32,
        lastblock1 as u32,
        lastblock2 as u32,
        lastblock3 as u32,
        lastblock4 as u32,
        lastblock5 as u32,
        lastblock6 as u32,
        lastblock7 as u32,
    );
    let lastnode = load_256_from_8xu32(
        lastnode0 as u32,
        lastnode1 as u32,
        lastnode2 as u32,
        lastnode3 as u32,
        lastnode4 as u32,
        lastnode5 as u32,
        lastnode6 as u32,
        lastnode7 as u32,
    );

    compress8_inner_inline(
        &mut h_vecs,
        msg0,
        msg1,
        msg2,
        msg3,
        msg4,
        msg5,
        msg6,
        msg7,
        count_low,
        count_high,
        lastblock,
        lastnode,
    );

    let deinterleaved = interleave_vecs(
        h_vecs[0], h_vecs[1], h_vecs[2], h_vecs[3], h_vecs[4], h_vecs[5], h_vecs[6], h_vecs[7],
    );
    *h0 = mem::transmute(deinterleaved[0]);
    *h1 = mem::transmute(deinterleaved[1]);
    *h2 = mem::transmute(deinterleaved[2]);
    *h3 = mem::transmute(deinterleaved[3]);
    *h4 = mem::transmute(deinterleaved[4]);
    *h5 = mem::transmute(deinterleaved[5]);
    *h6 = mem::transmute(deinterleaved[6]);
    *h7 = mem::transmute(deinterleaved[7]);
}

#[inline(always)]
unsafe fn interleave128(a: __m256i, b: __m256i) -> (__m256i, __m256i) {
    (
        _mm256_permute2x128_si256(a, b, 0x20),
        _mm256_permute2x128_si256(a, b, 0x31),
    )
}

#[cfg(test)]
fn cast_out(x: __m256i) -> [u32; 8] {
    unsafe { mem::transmute(x) }
}

#[cfg(test)]
#[test]
fn test_interleave128() {
    unsafe {
        let a = load_256_from_8xu32(10, 11, 12, 13, 14, 15, 16, 17);
        let b = load_256_from_8xu32(20, 21, 22, 23, 24, 25, 26, 27);

        let expected_a = load_256_from_8xu32(10, 11, 12, 13, 20, 21, 22, 23);
        let expected_b = load_256_from_8xu32(14, 15, 16, 17, 24, 25, 26, 27);

        let (out_a, out_b) = interleave128(a, b);

        assert_eq!(cast_out(expected_a), cast_out(out_a));
        assert_eq!(cast_out(expected_b), cast_out(out_b));
    }
}

#[inline(always)]
unsafe fn load_2x256(msg: &[u8; BLOCKBYTES]) -> (__m256i, __m256i) {
    (
        _mm256_loadu_si256(msg.as_ptr() as *const __m256i),
        _mm256_loadu_si256((msg.as_ptr() as *const __m256i).add(1)),
    )
}

#[cfg(test)]
#[test]
fn test_load_2x256() {
    unsafe {
        let input: [u64; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
        let input_bytes: [u8; BLOCKBYTES] = mem::transmute(input);
        let (out_a, out_b) = load_2x256(&input_bytes);

        let expected_a = load_256_from_8xu32(0, 0, 1, 0, 2, 0, 3, 0);
        let expected_b = load_256_from_8xu32(4, 0, 5, 0, 6, 0, 7, 0);

        assert_eq!(cast_out(expected_a), cast_out(out_a));
        assert_eq!(cast_out(expected_b), cast_out(out_b));
    }
}

#[inline(always)]
unsafe fn interleave_vecs(
    vec_a: __m256i,
    vec_b: __m256i,
    vec_c: __m256i,
    vec_d: __m256i,
    vec_e: __m256i,
    vec_f: __m256i,
    vec_g: __m256i,
    vec_h: __m256i,
) -> [__m256i; 8] {
    // Interleave 32-bit lanes. The low unpack is lanes 00/11/44/55, and the high is 22/33/66/77.
    let ab_0145 = _mm256_unpacklo_epi32(vec_a, vec_b);
    let ab_2367 = _mm256_unpackhi_epi32(vec_a, vec_b);
    let cd_0145 = _mm256_unpacklo_epi32(vec_c, vec_d);
    let cd_2367 = _mm256_unpackhi_epi32(vec_c, vec_d);
    let ef_0145 = _mm256_unpacklo_epi32(vec_e, vec_f);
    let ef_2367 = _mm256_unpackhi_epi32(vec_e, vec_f);
    let gh_0145 = _mm256_unpacklo_epi32(vec_g, vec_h);
    let gh_2367 = _mm256_unpackhi_epi32(vec_g, vec_h);

    // Interleave 64-bit lates. The low unpack is lanes 00/22 and the high is 11/33.
    let abcd_04 = _mm256_unpacklo_epi64(ab_0145, cd_0145);
    let abcd_15 = _mm256_unpackhi_epi64(ab_0145, cd_0145);
    let abcd_26 = _mm256_unpacklo_epi64(ab_2367, cd_2367);
    let abcd_37 = _mm256_unpackhi_epi64(ab_2367, cd_2367);
    let efgh_04 = _mm256_unpacklo_epi64(ef_0145, gh_0145);
    let efgh_15 = _mm256_unpackhi_epi64(ef_0145, gh_0145);
    let efgh_26 = _mm256_unpacklo_epi64(ef_2367, gh_2367);
    let efgh_37 = _mm256_unpackhi_epi64(ef_2367, gh_2367);

    // Interleave 128-bit lanes.
    let (abcdefg_0, abcdefg_4) = interleave128(abcd_04, efgh_04);
    let (abcdefg_1, abcdefg_5) = interleave128(abcd_15, efgh_15);
    let (abcdefg_2, abcdefg_6) = interleave128(abcd_26, efgh_26);
    let (abcdefg_3, abcdefg_7) = interleave128(abcd_37, efgh_37);

    [
        abcdefg_0, abcdefg_1, abcdefg_2, abcdefg_3, abcdefg_4, abcdefg_5, abcdefg_6, abcdefg_7,
    ]
}

#[cfg(test)]
#[test]
fn test_interleave_vecs() {
    unsafe {
        let vec_a = load_256_from_8xu32(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let vec_b = load_256_from_8xu32(0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17);
        let vec_c = load_256_from_8xu32(0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27);
        let vec_d = load_256_from_8xu32(0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37);
        let vec_e = load_256_from_8xu32(0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47);
        let vec_f = load_256_from_8xu32(0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57);
        let vec_g = load_256_from_8xu32(0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67);
        let vec_h = load_256_from_8xu32(0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77);

        let expected_a = load_256_from_8xu32(0x00, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70);
        let expected_b = load_256_from_8xu32(0x01, 0x11, 0x21, 0x31, 0x41, 0x51, 0x61, 0x71);
        let expected_c = load_256_from_8xu32(0x02, 0x12, 0x22, 0x32, 0x42, 0x52, 0x62, 0x72);
        let expected_d = load_256_from_8xu32(0x03, 0x13, 0x23, 0x33, 0x43, 0x53, 0x63, 0x73);
        let expected_e = load_256_from_8xu32(0x04, 0x14, 0x24, 0x34, 0x44, 0x54, 0x64, 0x74);
        let expected_f = load_256_from_8xu32(0x05, 0x15, 0x25, 0x35, 0x45, 0x55, 0x65, 0x75);
        let expected_g = load_256_from_8xu32(0x06, 0x16, 0x26, 0x36, 0x46, 0x56, 0x66, 0x76);
        let expected_h = load_256_from_8xu32(0x07, 0x17, 0x27, 0x37, 0x47, 0x57, 0x67, 0x77);

        let interleaved = interleave_vecs(vec_a, vec_b, vec_c, vec_d, vec_e, vec_f, vec_g, vec_h);

        let [out_a, out_b, out_c, out_d, out_e, out_f, out_g, out_h] = interleaved;
        assert_eq!(cast_out(expected_a), cast_out(out_a));
        assert_eq!(cast_out(expected_b), cast_out(out_b));
        assert_eq!(cast_out(expected_c), cast_out(out_c));
        assert_eq!(cast_out(expected_d), cast_out(out_d));
        assert_eq!(cast_out(expected_e), cast_out(out_e));
        assert_eq!(cast_out(expected_f), cast_out(out_f));
        assert_eq!(cast_out(expected_g), cast_out(out_g));
        assert_eq!(cast_out(expected_h), cast_out(out_h));

        // Check that interleaving again undoes the operation.
        let deinterleaved = interleave_vecs(out_a, out_b, out_c, out_d, out_e, out_f, out_g, out_h);
        let [out2_a, out2_b, out2_c, out2_d, out2_e, out2_f, out2_g, out2_h] = deinterleaved;
        assert_eq!(cast_out(vec_a), cast_out(out2_a));
        assert_eq!(cast_out(vec_b), cast_out(out2_b));
        assert_eq!(cast_out(vec_c), cast_out(out2_c));
        assert_eq!(cast_out(vec_d), cast_out(out2_d));
        assert_eq!(cast_out(vec_e), cast_out(out2_e));
        assert_eq!(cast_out(vec_f), cast_out(out2_f));
        assert_eq!(cast_out(vec_g), cast_out(out2_g));
        assert_eq!(cast_out(vec_h), cast_out(out2_h));
    }
}

#[inline(always)]
pub unsafe fn load_msg_vecs_interleave(
    msg_a: &[u8; BLOCKBYTES],
    msg_b: &[u8; BLOCKBYTES],
    msg_c: &[u8; BLOCKBYTES],
    msg_d: &[u8; BLOCKBYTES],
    msg_e: &[u8; BLOCKBYTES],
    msg_f: &[u8; BLOCKBYTES],
    msg_g: &[u8; BLOCKBYTES],
    msg_h: &[u8; BLOCKBYTES],
) -> [__m256i; 16] {
    let (front_a, back_a) = load_2x256(msg_a);
    let (front_b, back_b) = load_2x256(msg_b);
    let (front_c, back_c) = load_2x256(msg_c);
    let (front_d, back_d) = load_2x256(msg_d);
    let (front_e, back_e) = load_2x256(msg_e);
    let (front_f, back_f) = load_2x256(msg_f);
    let (front_g, back_g) = load_2x256(msg_g);
    let (front_h, back_h) = load_2x256(msg_h);

    let front_interleaved = interleave_vecs(
        front_a, front_b, front_c, front_d, front_e, front_f, front_g, front_h,
    );
    let back_interleaved = interleave_vecs(
        back_a, back_b, back_c, back_d, back_e, back_f, back_g, back_h,
    );

    [
        front_interleaved[0],
        front_interleaved[1],
        front_interleaved[2],
        front_interleaved[3],
        front_interleaved[4],
        front_interleaved[5],
        front_interleaved[6],
        front_interleaved[7],
        back_interleaved[0],
        back_interleaved[1],
        back_interleaved[2],
        back_interleaved[3],
        back_interleaved[4],
        back_interleaved[5],
        back_interleaved[6],
        back_interleaved[7],
    ]
}

#[inline(always)]
pub unsafe fn load_msg_vecs_gather(
    msg_0: &[u8; BLOCKBYTES],
    msg_1: &[u8; BLOCKBYTES],
    msg_2: &[u8; BLOCKBYTES],
    msg_3: &[u8; BLOCKBYTES],
    msg_4: &[u8; BLOCKBYTES],
    msg_5: &[u8; BLOCKBYTES],
    msg_6: &[u8; BLOCKBYTES],
    msg_7: &[u8; BLOCKBYTES],
) -> [__m256i; 16] {
    let mut buf = [0i32; 8 * 16];
    {
        let refs = mut_array_refs!(&mut buf, 16, 16, 16, 16, 16, 16, 16, 16);
        *refs.0 = mem::transmute(*msg_0);
        *refs.1 = mem::transmute(*msg_1);
        *refs.2 = mem::transmute(*msg_2);
        *refs.3 = mem::transmute(*msg_3);
        *refs.4 = mem::transmute(*msg_4);
        *refs.5 = mem::transmute(*msg_5);
        *refs.6 = mem::transmute(*msg_6);
        *refs.7 = mem::transmute(*msg_7);
    }

    let indexes = load_256_from_8xu32(
        0 * BLOCKBYTES as u32,
        1 * BLOCKBYTES as u32,
        2 * BLOCKBYTES as u32,
        3 * BLOCKBYTES as u32,
        4 * BLOCKBYTES as u32,
        5 * BLOCKBYTES as u32,
        6 * BLOCKBYTES as u32,
        7 * BLOCKBYTES as u32,
    );
    [
        _mm256_i32gather_epi32(buf.as_ptr().add(0), indexes, 1),
        _mm256_i32gather_epi32(buf.as_ptr().add(1), indexes, 1),
        _mm256_i32gather_epi32(buf.as_ptr().add(2), indexes, 1),
        _mm256_i32gather_epi32(buf.as_ptr().add(3), indexes, 1),
        _mm256_i32gather_epi32(buf.as_ptr().add(4), indexes, 1),
        _mm256_i32gather_epi32(buf.as_ptr().add(5), indexes, 1),
        _mm256_i32gather_epi32(buf.as_ptr().add(6), indexes, 1),
        _mm256_i32gather_epi32(buf.as_ptr().add(7), indexes, 1),
        _mm256_i32gather_epi32(buf.as_ptr().add(8), indexes, 1),
        _mm256_i32gather_epi32(buf.as_ptr().add(9), indexes, 1),
        _mm256_i32gather_epi32(buf.as_ptr().add(10), indexes, 1),
        _mm256_i32gather_epi32(buf.as_ptr().add(11), indexes, 1),
        _mm256_i32gather_epi32(buf.as_ptr().add(12), indexes, 1),
        _mm256_i32gather_epi32(buf.as_ptr().add(13), indexes, 1),
        _mm256_i32gather_epi32(buf.as_ptr().add(14), indexes, 1),
        _mm256_i32gather_epi32(buf.as_ptr().add(15), indexes, 1),
    ]
}

#[cfg(test)]
#[test]
fn test_gather() {
    unsafe {
        let mut input = [0u32; 8 * 16];
        for i in 0..8 * 16 {
            input[i] = i as u32;
        }
        let input_bytes: [u8; 8 * BLOCKBYTES] = mem::transmute(input);
        let blocks = array_refs!(
            &input_bytes,
            BLOCKBYTES,
            BLOCKBYTES,
            BLOCKBYTES,
            BLOCKBYTES,
            BLOCKBYTES,
            BLOCKBYTES,
            BLOCKBYTES,
            BLOCKBYTES
        );

        let expected_vecs = load_msg_vecs_naive(
            blocks.0, blocks.1, blocks.2, blocks.3, blocks.4, blocks.5, blocks.6, blocks.7,
        );

        let gather_vecs = load_msg_vecs_gather(
            blocks.0, blocks.1, blocks.2, blocks.3, blocks.4, blocks.5, blocks.6, blocks.7,
        );

        for i in 0..expected_vecs.len() {
            println!("{}", i);
            assert_eq!(cast_out(expected_vecs[i]), cast_out(gather_vecs[i]));
        }
    }
}

#[target_feature(enable = "avx2")]
pub unsafe fn compress8_inner(
    h_vecs: &mut [__m256i; 8],
    msg0: &Block,
    msg1: &Block,
    msg2: &Block,
    msg3: &Block,
    msg4: &Block,
    msg5: &Block,
    msg6: &Block,
    msg7: &Block,
    count_low: __m256i,
    count_high: __m256i,
    lastblock: __m256i,
    lastnode: __m256i,
) {
    compress8_inner_inline(
        h_vecs, msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, count_low, count_high, lastblock,
        lastnode,
    );
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn compress8_transposed(
    state_bytes: &mut Aligned8x8Words,
    msg0: &Block,
    msg1: &Block,
    msg2: &Block,
    msg3: &Block,
    msg4: &Block,
    msg5: &Block,
    msg6: &Block,
    msg7: &Block,
    count: [u64; 8],
    lastblock: [u32; 8],
    lastnode: [u32; 8],
) {
    // The 32-byte alignment of Aligned8x8Words makes this safe.
    let h_vecs = &mut *(state_bytes as *mut Aligned8x8Words as *mut [__m256i; 8]);
    compress8_inner_inline(
        h_vecs,
        msg0,
        msg1,
        msg2,
        msg3,
        msg4,
        msg5,
        msg6,
        msg7,
        load_256_from_8xu32(
            count[0] as u32,
            count[1] as u32,
            count[2] as u32,
            count[3] as u32,
            count[4] as u32,
            count[5] as u32,
            count[6] as u32,
            count[7] as u32,
        ),
        load_256_from_8xu32(
            (count[0] >> 32) as u32,
            (count[1] >> 32) as u32,
            (count[2] >> 32) as u32,
            (count[3] >> 32) as u32,
            (count[4] >> 32) as u32,
            (count[5] >> 32) as u32,
            (count[6] >> 32) as u32,
            (count[7] >> 32) as u32,
        ),
        mem::transmute(lastblock),
        mem::transmute(lastnode),
    );
}

#[inline(always)]
unsafe fn compress8_inner_inline(
    h_vecs: &mut [__m256i; 8],
    msg0: &Block,
    msg1: &Block,
    msg2: &Block,
    msg3: &Block,
    msg4: &Block,
    msg5: &Block,
    msg6: &Block,
    msg7: &Block,
    count_low: __m256i,
    count_high: __m256i,
    lastblock: __m256i,
    lastnode: __m256i,
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
        load_256_from_u32(IV[0]),
        load_256_from_u32(IV[1]),
        load_256_from_u32(IV[2]),
        load_256_from_u32(IV[3]),
        xor(load_256_from_u32(IV[4]), count_low),
        xor(load_256_from_u32(IV[5]), count_high),
        xor(load_256_from_u32(IV[6]), lastblock),
        xor(load_256_from_u32(IV[7]), lastnode),
    ];

    let msg_vecs = load_msg_vecs_interleave(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7);

    blake2s_round_8x(&mut v, &msg_vecs, 0);
    blake2s_round_8x(&mut v, &msg_vecs, 1);
    blake2s_round_8x(&mut v, &msg_vecs, 2);
    blake2s_round_8x(&mut v, &msg_vecs, 3);
    blake2s_round_8x(&mut v, &msg_vecs, 4);
    blake2s_round_8x(&mut v, &msg_vecs, 5);
    blake2s_round_8x(&mut v, &msg_vecs, 6);
    blake2s_round_8x(&mut v, &msg_vecs, 7);
    blake2s_round_8x(&mut v, &msg_vecs, 8);
    blake2s_round_8x(&mut v, &msg_vecs, 9);

    h_vecs[0] = xor(xor(h_vecs[0], v[0]), v[8]);
    h_vecs[1] = xor(xor(h_vecs[1], v[1]), v[9]);
    h_vecs[2] = xor(xor(h_vecs[2], v[2]), v[10]);
    h_vecs[3] = xor(xor(h_vecs[3], v[3]), v[11]);
    h_vecs[4] = xor(xor(h_vecs[4], v[4]), v[12]);
    h_vecs[5] = xor(xor(h_vecs[5], v[5]), v[13]);
    h_vecs[6] = xor(xor(h_vecs[6], v[6]), v[14]);
    h_vecs[7] = xor(xor(h_vecs[7], v[7]), v[15]);
}

#[inline(always)]
unsafe fn export_hashes(h_vecs: &[__m256i; 8], hash_length: u8) -> [Hash; 8] {
    // Interleave is its own inverse.
    let deinterleaved = interleave_vecs(
        h_vecs[0], h_vecs[1], h_vecs[2], h_vecs[3], h_vecs[4], h_vecs[5], h_vecs[6], h_vecs[7],
    );
    [
        Hash {
            len: hash_length,
            bytes: mem::transmute(deinterleaved[0]),
        },
        Hash {
            len: hash_length,
            bytes: mem::transmute(deinterleaved[1]),
        },
        Hash {
            len: hash_length,
            bytes: mem::transmute(deinterleaved[2]),
        },
        Hash {
            len: hash_length,
            bytes: mem::transmute(deinterleaved[3]),
        },
        Hash {
            len: hash_length,
            bytes: mem::transmute(deinterleaved[4]),
        },
        Hash {
            len: hash_length,
            bytes: mem::transmute(deinterleaved[5]),
        },
        Hash {
            len: hash_length,
            bytes: mem::transmute(deinterleaved[6]),
        },
        Hash {
            len: hash_length,
            bytes: mem::transmute(deinterleaved[7]),
        },
    ]
}

#[target_feature(enable = "avx2")]
pub unsafe fn blake2s_8way(
    // TODO: Separate params for each input.
    params: &Params,
    input0: &[u8],
    input1: &[u8],
    input2: &[u8],
    input3: &[u8],
    input4: &[u8],
    input5: &[u8],
    input6: &[u8],
    input7: &[u8],
) -> [Hash; 8] {
    let len = input0.len();
    let same_length = input1.len() == len
        && input2.len() == len
        && input3.len() == len
        && input4.len() == len
        && input5.len() == len
        && input6.len() == len
        && input7.len() == len;
    let even_length = len % BLOCKBYTES == 0;
    let nonempty = len != 0;
    assert!(
        same_length && even_length && nonempty,
        "invalid blake2s_8way inputs"
    );

    let param_words = params.make_words();
    let mut h_vecs = [
        load_256_from_u32(param_words[0]),
        load_256_from_u32(param_words[1]),
        load_256_from_u32(param_words[2]),
        load_256_from_u32(param_words[3]),
        load_256_from_u32(param_words[4]),
        load_256_from_u32(param_words[5]),
        load_256_from_u32(param_words[6]),
        load_256_from_u32(param_words[7]),
    ];
    let mut count = 0;

    loop {
        let msg0 = &*(input0.as_ptr().add(count) as *const Block);
        let msg1 = &*(input1.as_ptr().add(count) as *const Block);
        let msg2 = &*(input2.as_ptr().add(count) as *const Block);
        let msg3 = &*(input3.as_ptr().add(count) as *const Block);
        let msg4 = &*(input4.as_ptr().add(count) as *const Block);
        let msg5 = &*(input5.as_ptr().add(count) as *const Block);
        let msg6 = &*(input6.as_ptr().add(count) as *const Block);
        let msg7 = &*(input7.as_ptr().add(count) as *const Block);
        count += BLOCKBYTES;
        let count_low = load_256_from_u32(count as u32);
        let count_high = load_256_from_u32((count as u64 >> 32) as u32);
        let lastblock = load_256_from_u32(if count == len { !0 } else { 0 });
        let lastnode = load_256_from_u32(if params.last_node && count == len {
            !0
        } else {
            0
        });
        compress8_inner_inline(
            &mut h_vecs,
            msg0,
            msg1,
            msg2,
            msg3,
            msg4,
            msg5,
            msg6,
            msg7,
            count_low,
            count_high,
            lastblock,
            lastnode,
        );
        if count == len {
            return export_hashes(&h_vecs, params.hash_length);
        }
    }
}
