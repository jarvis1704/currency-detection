package com.biprangshu.currency.recognition

import org.junit.Assert.assertEquals
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder

class FramePreprocessorTest {

    private val output: ByteBuffer = ByteBuffer
        .allocateDirect(MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * MODEL_INPUT_CHANNELS * Float.SIZE_BYTES)
        .order(ByteOrder.nativeOrder())

    private fun frame(
        width: Int,
        height: Int,
        rowStride: Int = width * 4,
        pixel: (x: Int, y: Int) -> Triple<Int, Int, Int>,
    ): ByteBuffer {
        val buffer = ByteBuffer.allocate(rowStride * height)
        for (y in 0 until height) for (x in 0 until width) {
            val (r, g, b) = pixel(x, y)
            val index = y * rowStride + x * 4
            buffer.put(index, r.toByte())
            buffer.put(index + 1, g.toByte())
            buffer.put(index + 2, b.toByte())
            buffer.put(index + 3, 0xFF.toByte())
        }
        return buffer
    }

    private fun outputPixel(x: Int, y: Int): List<Float> {
        val base = (y * MODEL_INPUT_SIZE + x) * MODEL_INPUT_CHANNELS * Float.SIZE_BYTES
        return List(MODEL_INPUT_CHANNELS) { output.getFloat(base + it * Float.SIZE_BYTES) }
    }

    private fun assertPixel(expected: List<Float>, x: Int, y: Int) {
        val actual = outputPixel(x, y)
        expected.indices.forEach { assertEquals("channel $it at ($x,$y)", expected[it], actual[it], 0.01f) }
    }

    @Test
    fun `keeps rgb order and the 0 to 255 range without normalising`() {
        val rgba = frame(320, 320) { _, _ -> Triple(255, 128, 0) }
        writeModelInput(rgba, 320, 320, 320 * 4, output)
        assertPixel(listOf(255f, 128f, 0f), 0, 0)
        assertPixel(listOf(255f, 128f, 0f), 223, 223)
    }

    @Test
    fun `leaves the output ready to read and fully written`() {
        val rgba = frame(224, 224) { _, _ -> Triple(1, 2, 3) }
        writeModelInput(rgba, 224, 224, 224 * 4, output)
        assertEquals(0, output.position())
        assertPixel(listOf(1f, 2f, 3f), 223, 223)
    }

    @Test
    fun `source already at model size is copied exactly`() {
        val rgba = frame(224, 224) { x, y -> Triple(x, y, (x + y) / 2) }
        writeModelInput(rgba, 224, 224, 224 * 4, output)
        assertPixel(listOf(0f, 0f, 0f), 0, 0)
        assertPixel(listOf(17f, 200f, 108f), 17, 200)
        assertPixel(listOf(223f, 223f, 223f), 223, 223)
    }

    @Test
    fun `downscaling by two averages neighbours using half pixel centres`() {
        val rgba = frame(448, 448) { x, _ -> Triple(if (x % 2 == 0) 100 else 200, 0, 0) }
        writeModelInput(rgba, 448, 448, 448 * 4, output)
        assertPixel(listOf(150f, 0f, 0f), 100, 100)
    }

    @Test
    fun `wide frame is centre cropped`() {
        val rgba = frame(640, 480) { x, _ ->
            if (x in 80 until 560) Triple(50, 50, 50) else Triple(255, 0, 0)
        }
        writeModelInput(rgba, 640, 480, 640 * 4, output)
        assertPixel(listOf(50f, 50f, 50f), 0, 112)
        assertPixel(listOf(50f, 50f, 50f), 223, 112)
    }

    @Test
    fun `tall frame is centre cropped`() {
        val rgba = frame(480, 640) { _, y ->
            if (y in 80 until 560) Triple(50, 50, 50) else Triple(255, 0, 0)
        }
        writeModelInput(rgba, 480, 640, 480 * 4, output)
        assertPixel(listOf(50f, 50f, 50f), 112, 0)
        assertPixel(listOf(50f, 50f, 50f), 112, 223)
    }

    @Test
    fun `row padding is skipped`() {
        val rgba = frame(224, 224, rowStride = 256 * 4) { x, y -> Triple(x, y, 0) }
        writeModelInput(rgba, 224, 224, 256 * 4, output)
        assertPixel(listOf(40f, 90f, 0f), 40, 90)
    }
}
