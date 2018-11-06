#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use byteorder::{ByteOrder, LittleEndian};
use core::mem;

use crate::Block;
use crate::StateWords;
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
unsafe fn load_msg8_words(
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

#[inline(always)]
unsafe fn export_state_words_8x(
    orig_vec: __m256i,
    low_state: __m256i,
    high_state: __m256i,
    h0: &mut StateWords,
    h1: &mut StateWords,
    h2: &mut StateWords,
    h3: &mut StateWords,
    h4: &mut StateWords,
    h5: &mut StateWords,
    h6: &mut StateWords,
    h7: &mut StateWords,
    i: usize,
) {
    let parts: [u32; 8] = mem::transmute(xor(xor(orig_vec, low_state), high_state));
    h0[i] = parts[0];
    h1[i] = parts[1];
    h2[i] = parts[2];
    h3[i] = parts[3];
    h4[i] = parts[4];
    h5[i] = parts[5];
    h6[i] = parts[6];
    h7[i] = parts[7];
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
    let h_vecs = [
        load_256_from_8xu32(h0[0], h1[0], h2[0], h3[0], h4[0], h5[0], h6[0], h7[0]),
        load_256_from_8xu32(h0[1], h1[1], h2[1], h3[1], h4[1], h5[1], h6[1], h7[1]),
        load_256_from_8xu32(h0[2], h1[2], h2[2], h3[2], h4[2], h5[2], h6[2], h7[2]),
        load_256_from_8xu32(h0[3], h1[3], h2[3], h3[3], h4[3], h5[3], h6[3], h7[3]),
        load_256_from_8xu32(h0[4], h1[4], h2[4], h3[4], h4[4], h5[4], h6[4], h7[4]),
        load_256_from_8xu32(h0[5], h1[5], h2[5], h3[5], h4[5], h5[5], h6[5], h7[5]),
        load_256_from_8xu32(h0[6], h1[6], h2[6], h3[6], h4[6], h5[6], h6[6], h7[6]),
        load_256_from_8xu32(h0[7], h1[7], h2[7], h3[7], h4[7], h5[7], h6[7], h7[7]),
    ];
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
    let m = [
        load_msg8_words(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, 0),
        load_msg8_words(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, 1),
        load_msg8_words(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, 2),
        load_msg8_words(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, 3),
        load_msg8_words(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, 4),
        load_msg8_words(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, 5),
        load_msg8_words(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, 6),
        load_msg8_words(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, 7),
        load_msg8_words(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, 8),
        load_msg8_words(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, 9),
        load_msg8_words(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, 10),
        load_msg8_words(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, 11),
        load_msg8_words(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, 12),
        load_msg8_words(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, 13),
        load_msg8_words(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, 14),
        load_msg8_words(msg0, msg1, msg2, msg3, msg4, msg5, msg6, msg7, 15),
    ];

    blake2s_round_8x(&mut v, &m, 0);
    blake2s_round_8x(&mut v, &m, 1);
    blake2s_round_8x(&mut v, &m, 2);
    blake2s_round_8x(&mut v, &m, 3);
    blake2s_round_8x(&mut v, &m, 4);
    blake2s_round_8x(&mut v, &m, 5);
    blake2s_round_8x(&mut v, &m, 6);
    blake2s_round_8x(&mut v, &m, 7);
    blake2s_round_8x(&mut v, &m, 8);
    blake2s_round_8x(&mut v, &m, 9);

    export_state_words_8x(h_vecs[0], v[0], v[8], h0, h1, h2, h3, h4, h5, h6, h7, 0);
    export_state_words_8x(h_vecs[1], v[1], v[9], h0, h1, h2, h3, h4, h5, h6, h7, 1);
    export_state_words_8x(h_vecs[2], v[2], v[10], h0, h1, h2, h3, h4, h5, h6, h7, 2);
    export_state_words_8x(h_vecs[3], v[3], v[11], h0, h1, h2, h3, h4, h5, h6, h7, 3);
    export_state_words_8x(h_vecs[4], v[4], v[12], h0, h1, h2, h3, h4, h5, h6, h7, 4);
    export_state_words_8x(h_vecs[5], v[5], v[13], h0, h1, h2, h3, h4, h5, h6, h7, 5);
    export_state_words_8x(h_vecs[6], v[6], v[14], h0, h1, h2, h3, h4, h5, h6, h7, 6);
    export_state_words_8x(h_vecs[7], v[7], v[15], h0, h1, h2, h3, h4, h5, h6, h7, 7);
}
