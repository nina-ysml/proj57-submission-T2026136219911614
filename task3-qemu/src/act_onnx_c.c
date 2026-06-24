#include <onnxruntime_c_api.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>



static const OrtApi* g_ort = NULL;

static void fail(const char* msg) {
    fprintf(stderr, "%s\n", msg);
    exit(1);
}

static void check_ort(OrtStatus* status, const char* step) {
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "ORT error at %s: %s\n", step, msg);
        g_ort->ReleaseStatus(status);
        exit(1);
    }
}

static void make_path(char* out, size_t out_size, const char* a, const char* b) {
    int n = snprintf(out, out_size, "%s/%s", a, b);
    if (n < 0 || (size_t)n >= out_size) {
        fail("path too long");
    }
}

static float* read_f32_file(const char* path, size_t expected_count) {
    FILE* fp = fopen(path, "rb");
    if (fp == NULL) {
        fprintf(stderr, "failed to open: %s\n", path);
        exit(1);
    }

    size_t bytes = expected_count * sizeof(float);
    float* data = (float*)malloc(bytes);
    if (data == NULL) {
        fprintf(stderr, "malloc failed: %zu bytes\n", bytes);
        fclose(fp);
        exit(1);
    }

    size_t got = fread(data, 1, bytes, fp);
    fclose(fp);
    if (got != bytes) {
        fprintf(stderr, "unexpected file size: %s, got %zu, expected %zu\n", path, got, bytes);
        free(data);
        exit(1);
    }

    return data;
}

static void print_shape(OrtValue* value) {
    OrtTensorTypeAndShapeInfo* info = NULL;
    check_ort(g_ort->GetTensorTypeAndShape(value, &info), "GetTensorTypeAndShape");

    size_t rank = 0;
    check_ort(g_ort->GetDimensionsCount(info, &rank), "GetDimensionsCount");

    int64_t dims[8];
    if (rank > 8) {
        g_ort->ReleaseTensorTypeAndShapeInfo(info);
        fail("rank too large");
    }

    check_ort(g_ort->GetDimensions(info, dims, rank), "GetDimensions");

    printf("ONNX C output shape: [");
    for (size_t i = 0; i < rank; ++i) {
        if (i > 0) {
            printf(", ");
        }
        printf("%ld", (long)dims[i]);
    }
    printf("]\n");

    g_ort->ReleaseTensorTypeAndShapeInfo(info);
}

int main(int argc, char** argv) {

    const char* package_dir = "/root/act";
    if (argc >= 2) {
        package_dir = argv[1];
    }

    char model_path[512];
    char bin_dir[512];
    char image_path[512];
    char state_path[512];
    char q01_path[512];
    char q99_path[512];

    make_path(model_path, sizeof(model_path), package_dir, "act.onnx");
    make_path(bin_dir, sizeof(bin_dir), package_dir, "input_bins");
    make_path(image_path, sizeof(image_path), bin_dir, "image_1x1x3x224x224.bin");
    make_path(state_path, sizeof(state_path), bin_dir, "state_1x2.bin");
    make_path(q01_path, sizeof(q01_path), bin_dir, "action_q01_3.bin");
    make_path(q99_path, sizeof(q99_path), bin_dir, "action_q99_3.bin");

    printf("ACT ONNX C inference start\n");
    printf("model: %s\n", model_path);

    float* image = read_f32_file(image_path, 1 * 1 * 3 * 224 * 224);
    float* state = read_f32_file(state_path, 1 * 2);
    float* action_q01 = read_f32_file(q01_path, 3);
    float* action_q99 = read_f32_file(q99_path, 3);

    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (g_ort == NULL) {
        fail("failed to get ORT API");
    }
    printf("ORT version: %s\n", OrtGetApiBase()->GetVersionString());

    OrtEnv* env = NULL;
    OrtSessionOptions* session_options = NULL;
    OrtSession* session = NULL;
    OrtMemoryInfo* memory_info = NULL;
    OrtValue* image_tensor = NULL;
    OrtValue* state_tensor = NULL;
    OrtValue* output_tensor = NULL;

    check_ort(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "act_c_infer", &env), "CreateEnv");
    check_ort(g_ort->CreateSessionOptions(&session_options), "CreateSessionOptions");
    check_ort(g_ort->SetIntraOpNumThreads(session_options, 1), "SetIntraOpNumThreads");
    check_ort(g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_ALL), "SetSessionGraphOptimizationLevel");
    check_ort(g_ort->CreateSession(env, model_path, session_options, &session), "CreateSession");

    check_ort(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info), "CreateCpuMemoryInfo");

    int64_t image_shape[] = {1, 1, 3, 224, 224};
    int64_t state_shape[] = {1, 2};

    check_ort(
        g_ort->CreateTensorWithDataAsOrtValue(
            memory_info,
            image,
            1 * 1 * 3 * 224 * 224 * sizeof(float),
            image_shape,
            5,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &image_tensor),
        "Create image tensor");

    check_ort(
        g_ort->CreateTensorWithDataAsOrtValue(
            memory_info,
            state,
            1 * 2 * sizeof(float),
            state_shape,
            2,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &state_tensor),
        "Create state tensor");

    const char* input_names[] = {"image", "state"};
    const OrtValue* input_values[] = {image_tensor, state_tensor};
    const char* output_names[] = {"action"};

    check_ort(
        g_ort->Run(
            session,
            NULL,
            input_names,
            input_values,
            2,
            output_names,
            1,
            &output_tensor),
        "Run");

    print_shape(output_tensor);

    float* action = NULL;
    check_ort(g_ort->GetTensorMutableData(output_tensor, (void**)&action), "GetTensorMutableData");

    const char* labels[] = {"left_vel", "right_vel", "gripper_target"};
    printf("first step: ");
    for (int i = 0; i < 3; ++i) {
        float denom = action_q99[i] - action_q01[i];
        if (denom == 0.0f) {
            denom = 1e-8f;
        }
        float real_value = (action[i] + 1.0f) / 2.0f * denom + action_q01[i];
        if (i > 0) {
            printf(" | ");
        }
        printf("%s=%+.6f", labels[i], real_value);
    }
    printf("\n");

    if (output_tensor != NULL) g_ort->ReleaseValue(output_tensor);
    if (state_tensor != NULL) g_ort->ReleaseValue(state_tensor);
    if (image_tensor != NULL) g_ort->ReleaseValue(image_tensor);
    if (memory_info != NULL) g_ort->ReleaseMemoryInfo(memory_info);
    if (session != NULL) g_ort->ReleaseSession(session);
    if (session_options != NULL) g_ort->ReleaseSessionOptions(session_options);
    if (env != NULL) g_ort->ReleaseEnv(env);

    free(action_q99);
    free(action_q01);
    free(state);
    free(image);

    return 0;
}
