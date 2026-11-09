package com.biprangshu.currency.recognition

import java.nio.ByteBuffer
import kotlin.math.floor

internal const val MODEL_INPUT_SIZE = 224
internal const val MODEL_INPUT_CHANNELS = 3

private const val RGBA_PIXEL_STRIDE = 4

/**
 * Writes an RGBA_8888 frame into [output] the way training loaded its images: centre square crop,
 * bilinear resize with half-pixel centres to 224×224, then float32 R,G,B in 0–255. Normalisation
 * happens inside the model, so the values must not be scaled here.
 *
 * The frame is deliberately not rotated to match the device: training applied a random quarter
 * turn to every image, so the model already accepts a note at any quarter turn.
 */
internal fun writeModelInput(
    rgba: ByteBuffer,
    width: Int,
    height: Int,
    rowStride: Int,
    output: ByteBuffer,
) {
    val side = minOf(width, height)
    val cropLeft = (width - side) / 2
    val cropTop = (height - side) / 2
    val scale = side.toFloat() / MODEL_INPUT_SIZE

    fun channel(x: Int, y: Int, offset: Int): Int {
        val index = (cropTop + y) * rowStride + (cropLeft + x) * RGBA_PIXEL_STRIDE + offset
        return rgba.get(index).toInt() and 0xFF
    }

    output.rewind()
    for (outY in 0 until MODEL_INPUT_SIZE) {
        val sourceY = (outY + 0.5f) * scale - 0.5f
        val y0 = floor(sourceY).toInt().coerceIn(0, side - 1)
        val y1 = (y0 + 1).coerceAtMost(side - 1)
        val yWeight = (sourceY - y0).coerceIn(0f, 1f)

        for (outX in 0 until MODEL_INPUT_SIZE) {
            val sourceX = (outX + 0.5f) * scale - 0.5f
            val x0 = floor(sourceX).toInt().coerceIn(0, side - 1)
            val x1 = (x0 + 1).coerceAtMost(side - 1)
            val xWeight = (sourceX - x0).coerceIn(0f, 1f)

            for (offset in 0 until MODEL_INPUT_CHANNELS) {
                val top = lerp(channel(x0, y0, offset), channel(x1, y0, offset), xWeight)
                val bottom = lerp(channel(x0, y1, offset), channel(x1, y1, offset), xWeight)
                output.putFloat(top + (bottom - top) * yWeight)
            }
        }
    }
    output.rewind()
}

private fun lerp(from: Int, to: Int, weight: Float): Float = from + (to - from) * weight
