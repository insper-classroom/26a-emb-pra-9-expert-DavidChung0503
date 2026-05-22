#include "FreeRTOS.h"
#include "ei_accelerometer.h"
#include "ei_analogsensor.h"
#include "ei_at_handlers.h"
#include "ei_classifier_porting.h"
#include "ei_device_raspberry_rp2xxx.h"
#include "ei_dht11sensor.h"
#include "ei_inertialsensor.h"
#include "ei_rp2xxx_internal_temperature.h"
#include "ei_run_impulse.h"
#include "ei_ultrasonicsensor.h"
#include "pico/multicore.h"
#include "pico/stdlib.h"
#include "task.h"
#include <stdio.h>
#include <time.h>
#include <string.h>

// imu
#include <hardware/gpio.h>
#include <hardware/i2c.h>
#include <hardware/uart.h>
#include <pico/stdio.h>

// freertos
#include <FreeRTOS.h>
#include <queue.h>
#include <semphr.h>
#include <stdlib.h>
#include <task.h>

#include "mpu6050.h"
#include "edge-impulse-sdk/classifier/ei_model_types.h"
#include "edge-impulse-sdk/dsp/numpy.hpp"
#include "model-parameters/model_metadata.h"

using namespace ei;

extern "C" EI_IMPULSE_ERROR
run_classifier(ei::signal_t *signal, ei_impulse_result_t *result, bool debug);

static bool debug_nn = false;

const int MPU_ADDRESS  = 0x68;
const int I2C_SDA_GPIO = 4;
const int I2C_SCL_GPIO = 5;

#define LED_R_PIN 7
#define LED_G_PIN 8
#define LED_B_PIN 9

static void rgb_init()
{
    gpio_init(LED_R_PIN); gpio_set_dir(LED_R_PIN, GPIO_OUT);
    gpio_init(LED_G_PIN); gpio_set_dir(LED_G_PIN, GPIO_OUT);
    gpio_init(LED_B_PIN); gpio_set_dir(LED_B_PIN, GPIO_OUT);
    gpio_put(LED_R_PIN, 1);
    gpio_put(LED_G_PIN, 1);
    gpio_put(LED_B_PIN, 1);
}

static void set_rgb_led(bool r, bool g, bool b)
{
    gpio_put(LED_R_PIN, !r);
    gpio_put(LED_G_PIN, !g);
    gpio_put(LED_B_PIN, !b);
}

static void mpu6050_init()
{
    i2c_init(i2c_default, 400 * 1000);
    gpio_set_function(I2C_SDA_GPIO, GPIO_FUNC_I2C);
    gpio_set_function(I2C_SCL_GPIO, GPIO_FUNC_I2C);
    gpio_pull_up(I2C_SDA_GPIO);
    gpio_pull_up(I2C_SCL_GPIO);

    uint8_t buf[] = { 0x6B, 0x00 };
    i2c_write_blocking(i2c_default, MPU_ADDRESS, buf, 2, false);
}

static void mpu6050_read_raw(int16_t accel[3], int16_t gyro[3], int16_t *temp)
{
    uint8_t buffer[14];

    uint8_t val = 0x3B;
    i2c_write_blocking(i2c_default, MPU_ADDRESS, &val, 1, true);
    i2c_read_blocking(i2c_default, MPU_ADDRESS, buffer, 14, false);

    for (int i = 0; i < 3; i++) {
        accel[i] = (buffer[i * 2] << 8 | buffer[(i * 2) + 1]);
    }

    *temp = buffer[6] << 8 | buffer[7];

    for (int i = 0; i < 3; i++) {
        gyro[i] = (buffer[8 + i * 2] << 8 | buffer[8 + (i * 2) + 1]);
    }
}

static void gesture_recognize_task(void *p)
{
    mpu6050_init();
    rgb_init();

    int16_t accelerometer[3], gyro[3], temp;

    while (true) {
        float buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE] = { 0 };

        for (size_t ix = 0; ix < EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE; ix += 3) {
            mpu6050_read_raw(accelerometer, gyro, &temp);
            buffer[ix + 0] = accelerometer[0];
            buffer[ix + 1] = accelerometer[1];
            buffer[ix + 2] = accelerometer[2];

            vTaskDelay(pdMS_TO_TICKS(10));
        }

        ei::signal_t signal;
        int err = numpy::signal_from_buffer(buffer, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &signal);
        if (err != 0) {
            ei_printf("Failed to create signal from buffer (%d)\n", err);
            break;
        }

        ei_impulse_result_t result = { 0 };
        err = run_classifier(&signal, &result, debug_nn);
        if (err != EI_IMPULSE_OK) {
            ei_printf("ERR: Failed to run classifier (%d)\n", err);
            break;
        }

            result.timing.dsp,
            result.timing.classification,
            result.timing.anomaly);

        for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
            ei_printf("  %s: %.5f\n",
                result.classification[ix].label,
                result.classification[ix].value);
        }

#if EI_CLASSIFIER_HAS_ANOMALY == 1
        ei_printf("  anomaly score: %.3f\n", result.anomaly);
#endif

        int best_ix = 0;
        for (size_t ix = 1; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
            if (result.classification[ix].value > result.classification[best_ix].value)
                best_ix = (int)ix;
        }

        const char *label = result.classification[best_ix].label;
        ei_printf(">> Classe detectada: %s (%.2f)\n", label, result.classification[best_ix].value);

        if      (strcmp(label, "idle")   == 0) set_rgb_led(0, 0, 1); 
        else if (strcmp(label, "waving")   == 0) set_rgb_led(0, 1, 0); 
        else if (strcmp(label, "updown") == 0) set_rgb_led(1, 0, 0); 
        else                                    set_rgb_led(0, 0, 0); 
    }
}

int main(void)
{
    stdio_init_all();

    xTaskCreate(gesture_recognize_task, "gesture_task", 8192, NULL, 1, NULL);
    vTaskStartScheduler();

    while (true)
        ;
}