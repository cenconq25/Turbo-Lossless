/*
 * Structured 12-bit packer for BF16 lossless compression.
 * Format: [4-bit exp_group][1-bit sign][7-bit mantissa] = 12 bits
 * Top 15 consecutive exponents → groups 1-15, group 0 = escape.
 * Decode: exponent = BaseExp + group (ONE integer add, no lookup table).
 */

#include <stdint.h>
#include <string.h>
#include <stdlib.h>

/*
 * Find the optimal BaseExp: the 15 consecutive exponents covering most weights.
 * Returns BaseExp (exponent = BaseExp + group, group 1-15).
 * exp_rmap_out[256]: maps exponent → group (0 = escape).
 */
int find_base_exp(
    const uint16_t* raw_bf16,
    int64_t num_values,
    uint8_t* exp_rmap_out)    /* [256] output: exponent → group code */
{
    int64_t counts[256] = {0};
    for (int64_t i = 0; i < num_values; i++) {
        uint8_t exp = (raw_bf16[i] >> 7) & 0xFF;
        counts[exp]++;
    }

    /* Find the window of 15 consecutive exponents with maximum coverage */
    int best_start = 0;
    int64_t best_sum = 0;
    for (int start = 0; start <= 241; start++) {
        int64_t sum = 0;
        for (int j = 0; j < 15; j++) sum += counts[start + j];
        if (sum > best_sum) { best_sum = sum; best_start = start; }
    }

    int base_exp = best_start - 1;  /* group 1 → exponent base_exp + 1 */

    memset(exp_rmap_out, 0, 256);  /* 0 = escape */
    for (int g = 1; g <= 15; g++) {
        exp_rmap_out[base_exp + g] = (uint8_t)g;
    }

    return base_exp;
}

/*
 * Pack BF16 values into structured 12-bit format with CSR escape handling.
 * Format: [4-bit group][1-bit sign][7-bit mantissa] = 12 bits per element.
 * When group=0 (escape), correct BF16 value stored in patch arrays.
 */
int64_t pack_structured12_csr(
    const uint16_t* raw_bf16,
    int64_t num_values,
    const uint8_t* exp_rmap,       /* [256] exponent → group */
    uint32_t* packed_out,
    int32_t* row_offsets_out,
    int32_t* patch_cols_out,
    int16_t* patch_correct_out,
    int16_t* patch_wrong_out,
    int32_t wrong_value,
    int32_t M,
    int32_t K)
{
    int64_t num_patches = 0;

    for (int32_t r = 0; r <= M; r++)
        row_offsets_out[r] = 0;

    for (int64_t i = 0; i < num_values; i++) {
        uint16_t val = raw_bf16[i];
        uint8_t sign = (val >> 15) & 1;
        uint8_t exp = (val >> 7) & 0xFF;
        uint8_t mantissa = val & 0x7F;
        uint8_t group = exp_rmap[exp];

        /* Pack: [group(4)][sign(1)][mantissa(7)] = 12 bits */
        uint32_t packed12 = ((uint32_t)group << 8) | ((uint32_t)sign << 7) | mantissa;

        int64_t bit_pos = i * 12;
        int word = (int)(bit_pos >> 5);
        int shift = (int)(bit_pos & 31);
        packed_out[word] |= (packed12 & 0xFFF) << shift;
        if (shift + 12 > 32)
            packed_out[word + 1] |= (packed12 & 0xFFF) >> (32 - shift);

        if (group == 0) {
            int32_t row = (int32_t)(i / K);
            row_offsets_out[row + 1]++;
            num_patches++;
        }
    }

    /* Prefix sum */
    for (int32_t r = 0; r < M; r++)
        row_offsets_out[r + 1] += row_offsets_out[r];

    /* Fill patch data */
    int32_t* tmp = (int32_t*)malloc((size_t)M * sizeof(int32_t));
    for (int32_t r = 0; r < M; r++)
        tmp[r] = row_offsets_out[r];

    for (int64_t i = 0; i < num_values; i++) {
        uint16_t val = raw_bf16[i];
        uint8_t exp = (val >> 7) & 0xFF;
        if (exp_rmap[exp] == 0) {
            int32_t row = (int32_t)(i / K);
            int32_t col = (int32_t)(i % K);
            int32_t pos = tmp[row]++;
            patch_cols_out[pos] = col;
            patch_correct_out[pos] = (int16_t)val;
            patch_wrong_out[pos] = (int16_t)wrong_value;
        }
    }

    free(tmp);
    return num_patches;
}
