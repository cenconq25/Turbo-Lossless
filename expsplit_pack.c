/*
 * Exponent-Split 12-bit packer for BF16 lossless compression.
 * Splits BF16 into sign(1) + exp_code(4) + mantissa(7) = 12 bits.
 * Top 15 exponents → codes 1-15, rare exponents → code 0 (escape).
 * Decode is pure ALU (no codebook LDS lookup).
 */

#include <stdint.h>
#include <string.h>
#include <stdlib.h>

/*
 * Build exponent table: find top 15 exponents from the weight tensor.
 * exp_table_out[16]: exp_table[0] = unused (escape sentinel), exp_table[1-15] = exponents
 * exp_rmap_out[256]: maps exponent → code (0 = escape)
 */
int build_exp_table(
    const uint16_t* raw_bf16,
    int64_t num_values,
    uint8_t* exp_table_out,   /* [16] output */
    uint8_t* exp_rmap_out)    /* [256] output: exponent → code */
{
    int64_t counts[256] = {0};
    for (int64_t i = 0; i < num_values; i++) {
        uint8_t exp = (raw_bf16[i] >> 7) & 0xFF;
        counts[exp]++;
    }

    /* Sort by frequency, pick top 15 */
    memset(exp_rmap_out, 0, 256);  /* 0 = escape */
    memset(exp_table_out, 0, 16);

    for (int code = 1; code <= 15; code++) {
        int best = -1;
        int64_t best_count = -1;
        for (int e = 0; e < 256; e++) {
            if (counts[e] > best_count && exp_rmap_out[e] == 0) {
                best = e;
                best_count = counts[e];
            }
        }
        if (best < 0 || best_count == 0) break;
        exp_table_out[code] = (uint8_t)best;
        exp_rmap_out[best] = (uint8_t)code;
    }
    return 15;
}

/*
 * Pack BF16 values into 12-bit exponent-split format with CSR escape handling.
 * Format: [1-bit sign][4-bit exp_code][7-bit mantissa] = 12 bits per element.
 * When exp_code=0 (escape), correct BF16 value stored in patch arrays.
 */
int64_t pack_expsplit_csr(
    const uint16_t* raw_bf16,
    int64_t num_values,
    const uint8_t* exp_rmap,      /* [256] exponent → code */
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

    /* First pass: pack 12-bit + count patches per row */
    for (int64_t i = 0; i < num_values; i++) {
        uint16_t val = raw_bf16[i];
        uint8_t sign = (val >> 15) & 1;
        uint8_t exp = (val >> 7) & 0xFF;
        uint8_t mantissa = val & 0x7F;
        uint8_t code = exp_rmap[exp];

        /* Pack: [sign(1)][exp_code(4)][mantissa(7)] = 12 bits */
        uint32_t packed12 = ((uint32_t)sign << 11) | ((uint32_t)code << 7) | mantissa;

        int64_t bit_pos = i * 12;
        int word = (int)(bit_pos >> 5);
        int shift = (int)(bit_pos & 31);
        packed_out[word] |= (packed12 & 0xFFF) << shift;
        if (shift + 12 > 32)
            packed_out[word + 1] |= (packed12 & 0xFFF) >> (32 - shift);

        if (code == 0) {
            int32_t row = (int32_t)(i / K);
            row_offsets_out[row + 1]++;
            num_patches++;
        }
    }

    /* Prefix sum */
    for (int32_t r = 0; r < M; r++)
        row_offsets_out[r + 1] += row_offsets_out[r];

    /* Second pass: fill patch data */
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
