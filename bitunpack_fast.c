/*
 * Fast bit unpacking for TLC decoder.
 * Compile: gcc -O3 -shared -fPIC -o bitunpack_fast.so bitunpack_fast.c
 */
#include <stdint.h>

/*
 * Unpack encoded values from a uint32 bitstream (MSB-first within each word).
 *
 * packed: uint32 bitstream
 * codebook: int16[7 * tier_size] frequency-ranked BF16 values
 * index_bits: bits per codebook index
 * tier_size: 2^index_bits
 * num_values: total values to decode
 * output: pre-allocated int16 buffer for decoded values
 *
 * Returns: total bits consumed
 */

static inline int read_bit_fast(const uint32_t *buf, uint64_t bit_pos) {
    int word_idx = bit_pos >> 5;
    int bit_idx = 31 - (bit_pos & 31);
    return (buf[word_idx] >> bit_idx) & 1;
}

static inline uint32_t read_bits_fast(const uint32_t *buf, uint64_t bit_pos, int num_bits) {
    uint32_t value = 0;
    while (num_bits > 0) {
        int word_idx = bit_pos >> 5;
        int bit_in_word = bit_pos & 31;
        int space = 32 - bit_in_word;
        int to_read = (space < num_bits) ? space : num_bits;
        int shift = space - to_read;
        uint32_t bits = (buf[word_idx] >> shift) & ((1U << to_read) - 1);
        value = (value << to_read) | bits;
        bit_pos += to_read;
        num_bits -= to_read;
    }
    return value;
}

uint64_t unpack_values(
    const uint32_t *packed,
    const int16_t *codebook,
    int index_bits,
    int tier_size,
    int64_t num_values,
    int16_t *output
) {
    uint64_t bit_pos = 0;

    for (int64_t i = 0; i < num_values; i++) {
        /* Count leading 1-bits (unary prefix) */
        int tier = 0;
        while (tier < 7) {
            if (read_bit_fast(packed, bit_pos) == 0) {
                bit_pos++;  /* consume the 0 */
                break;
            }
            tier++;
            bit_pos++;
        }

        if (tier < 7) {
            /* Coded value */
            uint32_t index = read_bits_fast(packed, bit_pos, index_bits);
            bit_pos += index_bits;
            output[i] = codebook[tier * tier_size + index];
        } else {
            /* Escape: 7 ones consumed, read 16 raw bits */
            uint32_t raw = read_bits_fast(packed, bit_pos, 16);
            bit_pos += 16;
            output[i] = (int16_t)raw;
        }
    }

    return bit_pos;
}
