/**
 * C-FFI smoke test for bloqade-lanes-bytecode.
 *
 * Exercises the core C API: parsing, serialization, validation, and error
 * handling. Returns 0 on success, non-zero on failure.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bloqade_lanes_bytecode.h"

#define ASSERT_OK(expr, msg)                                                   \
    do {                                                                        \
        enum BlqdStatus _s = (expr);                                           \
        if (_s != BLQD_STATUS_OK) {                                            \
            const char *_e = blqd_last_error();                                \
            fprintf(stderr, "FAIL: %s (status=%d, err=%s)\n", msg, _s,         \
                    _e ? _e : "(none)");                                        \
            return 1;                                                          \
        }                                                                      \
    } while (0)

#define ASSERT_EQ(a, b, msg)                                                   \
    do {                                                                        \
        if ((a) != (b)) {                                                      \
            fprintf(stderr, "FAIL: %s (expected %d, got %d)\n", msg,           \
                    (int)(b), (int)(a));                                        \
            return 1;                                                          \
        }                                                                      \
    } while (0)

#define ASSERT_TRUE(cond, msg)                                                 \
    do {                                                                        \
        if (!(cond)) {                                                         \
            fprintf(stderr, "FAIL: %s\n", msg);                                \
            return 1;                                                          \
        }                                                                      \
    } while (0)

/* Minimal valid SST program: two locations, initial_fill, halt.
 * LocationAddr (64-bit): [zone_id:8][word_id:16][site_id:16][pad:24]
 * zone 0, word 0, site 0 = 0x0000000000000000
 * zone 0, word 0, site 1 = 0x0000000001000000 */
static const char *VALID_PROGRAM =
    ".version 1\n"
    "const_loc 0x0000000000000000\n"
    "const_loc 0x0000000001000000\n"
    "initial_fill 2\n"
    "halt\n";

/* Program that triggers a structural error (initial_fill not first). */
static const char *INVALID_STRUCTURE =
    ".version 1\n"
    "halt\n"
    "const_loc 0x0000000000000000\n"
    "initial_fill 1\n";

/* Program that triggers a stack underflow (pop on empty stack). */
static const char *STACK_UNDERFLOW =
    ".version 1\n"
    "fill 1\n";

/* Program with type mismatch: fill expects locations, gets float. */
static const char *TYPE_MISMATCH =
    ".version 1\n"
    "const_loc 0x0000000000000000\n"
    "initial_fill 1\n"
    "const_float 3.14\n"
    "fill 1\n"
    "halt\n";

int main(void) {
    int tests_passed = 0;

    /* --- Test 1: Parse text -> Program handle --- */
    {
        struct BLQDProgram *prog = NULL;
        ASSERT_OK(blqd_program_from_text(VALID_PROGRAM, &prog),
                  "parse valid program");
        ASSERT_TRUE(prog != NULL, "program handle is non-null");

        uint32_t count = blqd_program_instruction_count(prog);
        ASSERT_EQ(count, 4, "instruction count");

        uint16_t major = 0, minor = 0;
        blqd_program_version(prog, &major, &minor);
        ASSERT_EQ(major, 1, "version major");

        blqd_program_free(prog);
        printf("  PASS: parse text to program\n");
        tests_passed++;
    }

    /* --- Test 2: Text -> binary -> text round-trip --- */
    {
        struct BLQDProgram *prog = NULL;
        ASSERT_OK(blqd_program_from_text(VALID_PROGRAM, &prog),
                  "parse for round-trip");

        /* Serialize to binary */
        uint8_t *bin_data = NULL;
        uintptr_t bin_len = 0;
        ASSERT_OK(blqd_program_to_binary(prog, &bin_data, &bin_len),
                  "serialize to binary");
        ASSERT_TRUE(bin_data != NULL && bin_len > 0, "binary data is non-empty");

        /* Parse binary back */
        struct BLQDProgram *prog2 = NULL;
        ASSERT_OK(blqd_program_from_binary(bin_data, bin_len, &prog2),
                  "parse binary");
        blqd_free_bytes(bin_data, bin_len);

        /* Convert back to text */
        char *text_out = NULL;
        ASSERT_OK(blqd_program_to_text(prog2, &text_out), "serialize to text");
        ASSERT_TRUE(text_out != NULL, "text output is non-null");
        ASSERT_TRUE(strstr(text_out, "initial_fill") != NULL,
                    "round-trip text contains initial_fill");

        blqd_free_string(text_out);
        blqd_program_free(prog);
        blqd_program_free(prog2);
        printf("  PASS: text -> binary -> text round-trip\n");
        tests_passed++;
    }

    /* --- Test 3: Structural validation (valid program) --- */
    {
        struct BLQDProgram *prog = NULL;
        ASSERT_OK(blqd_program_from_text(VALID_PROGRAM, &prog),
                  "parse for validation");

        struct BLQDValidationErrors *errs = NULL;
        ASSERT_OK(blqd_validate_structure(prog, &errs), "validate structure");

        uint32_t err_count = blqd_validation_errors_count(errs);
        ASSERT_EQ(err_count, 0, "no structural errors");

        blqd_validation_errors_free(errs);
        blqd_program_free(prog);
        printf("  PASS: structural validation (valid)\n");
        tests_passed++;
    }

    /* --- Test 4: Structural validation detects error --- */
    {
        struct BLQDProgram *prog = NULL;
        ASSERT_OK(blqd_program_from_text(INVALID_STRUCTURE, &prog),
                  "parse invalid-structure program");

        struct BLQDValidationErrors *errs = NULL;
        enum BlqdStatus s = blqd_validate_structure(prog, &errs);
        ASSERT_EQ(s, BLQD_STATUS_ERR_VALIDATION,
                  "structural validation returns ErrValidation");

        uint32_t err_count = blqd_validation_errors_count(errs);
        ASSERT_TRUE(err_count > 0, "at least one validation error");

        const char *msg = blqd_validation_error_message(errs, 0);
        ASSERT_TRUE(msg != NULL, "error message is non-null");
        ASSERT_TRUE(strstr(msg, "initial_fill") != NULL,
                    "error mentions initial_fill");

        /* Out-of-range index returns NULL */
        ASSERT_TRUE(blqd_validation_error_message(errs, 999) == NULL,
                    "out-of-range index returns NULL");

        blqd_validation_errors_free(errs);
        blqd_program_free(prog);
        printf("  PASS: structural validation (invalid)\n");
        tests_passed++;
    }

    /* --- Test 5: Stack simulation --- */
    {
        struct BLQDProgram *prog = NULL;
        ASSERT_OK(blqd_program_from_text(VALID_PROGRAM, &prog),
                  "parse for stack sim");

        struct BLQDValidationErrors *errs = NULL;
        ASSERT_OK(blqd_simulate_stack(prog, &errs), "simulate stack");

        uint32_t err_count = blqd_validation_errors_count(errs);
        ASSERT_EQ(err_count, 0, "no stack simulation errors");

        blqd_validation_errors_free(errs);
        blqd_program_free(prog);
        printf("  PASS: stack simulation\n");
        tests_passed++;
    }

    /* --- Test 6: Stack simulation detects underflow --- */
    {
        struct BLQDProgram *prog = NULL;
        ASSERT_OK(blqd_program_from_text(STACK_UNDERFLOW, &prog),
                  "parse stack-underflow program");

        struct BLQDValidationErrors *errs = NULL;
        enum BlqdStatus s = blqd_simulate_stack(prog, &errs);
        ASSERT_EQ(s, BLQD_STATUS_ERR_VALIDATION,
                  "stack sim returns ErrValidation for underflow");

        uint32_t err_count = blqd_validation_errors_count(errs);
        ASSERT_TRUE(err_count > 0, "at least one stack error");

        const char *msg = blqd_validation_error_message(errs, 0);
        ASSERT_TRUE(msg != NULL, "underflow error message is non-null");

        blqd_validation_errors_free(errs);
        blqd_program_free(prog);
        printf("  PASS: stack simulation (underflow)\n");
        tests_passed++;
    }

    /* --- Test 7: Stack simulation detects type mismatch --- */
    {
        struct BLQDProgram *prog = NULL;
        ASSERT_OK(blqd_program_from_text(TYPE_MISMATCH, &prog),
                  "parse type-mismatch program");

        struct BLQDValidationErrors *errs = NULL;
        enum BlqdStatus s = blqd_simulate_stack(prog, &errs);
        ASSERT_EQ(s, BLQD_STATUS_ERR_VALIDATION,
                  "stack sim returns ErrValidation for type mismatch");

        uint32_t err_count = blqd_validation_errors_count(errs);
        ASSERT_TRUE(err_count > 0, "at least one type mismatch error");

        const char *msg = blqd_validation_error_message(errs, 0);
        ASSERT_TRUE(msg != NULL, "type mismatch error message is non-null");
        ASSERT_TRUE(strstr(msg, "type mismatch") != NULL ||
                    strstr(msg, "Type mismatch") != NULL ||
                    strstr(msg, "TypeMismatch") != NULL,
                    "error mentions type mismatch");

        blqd_validation_errors_free(errs);
        blqd_program_free(prog);
        printf("  PASS: stack simulation (type mismatch)\n");
        tests_passed++;
    }

    /* --- Test 8: Null pointer safety --- */
    {
        enum BlqdStatus s = blqd_program_from_text(NULL, NULL);
        ASSERT_EQ(s, BLQD_STATUS_ERR_NULL_PTR, "null text returns null-ptr status");

        s = blqd_program_from_binary(NULL, 0, NULL);
        ASSERT_EQ(s, BLQD_STATUS_ERR_NULL_PTR, "null binary returns null-ptr status");

        /* Freeing NULL should be safe (no-op) */
        blqd_program_free(NULL);
        blqd_arch_free(NULL);
        blqd_validation_errors_free(NULL);

        printf("  PASS: null pointer safety\n");
        tests_passed++;
    }

    /* --- Test 9: blqd_last_error after failure --- */
    {
        struct BLQDProgram *prog = NULL;
        enum BlqdStatus s = blqd_program_from_text("not valid sst", &prog);
        ASSERT_TRUE(s != BLQD_STATUS_OK, "parse garbage fails");

        const char *err = blqd_last_error();
        ASSERT_TRUE(err != NULL, "last_error is set after failure");
        ASSERT_TRUE(strlen(err) > 0, "last_error is non-empty");

        printf("  PASS: last_error after failure\n");
        tests_passed++;
    }

    printf("\n=== C-FFI smoke test: %d tests passed ===\n", tests_passed);
    return 0;
}
