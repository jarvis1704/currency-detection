package com.biprangshu.currency.recognition

/**
 * An axis-aligned box in coordinates normalised to 0–1 across the camera frame, in the frame's own
 * (sensor) orientation, not the display's. Normalising keeps the box independent of the analysis
 * resolution so the UI can scale it onto the preview surface.
 */
data class NormalizedBox(val left: Float, val top: Float, val right: Float, val bottom: Float) {

    val centerX: Float get() = (left + right) / 2
    val centerY: Float get() = (top + bottom) / 2

    /**
     * Converts a box measured on the frame after it was turned upright by [rotationDegrees]
     * clockwise back into the frame's own orientation.
     */
    fun uprightToBuffer(rotationDegrees: Int): NormalizedBox = when (rotationDegrees) {
        0 -> this
        90 -> NormalizedBox(left = top, top = 1 - right, right = bottom, bottom = 1 - left)
        180 -> NormalizedBox(left = 1 - right, top = 1 - bottom, right = 1 - left, bottom = 1 - top)
        270 -> NormalizedBox(left = 1 - bottom, top = left, right = 1 - top, bottom = right)
        else -> throw IllegalArgumentException("Rotation must be a quarter turn: $rotationDegrees")
    }

    /**
     * Whether the box's centre lies in the centre square of a [width]×[height] frame, the only
     * region the classifier looks at (see [writeModelInput]). An object outside it cannot be the
     * note the classifier recognised.
     */
    fun isCenteredInSquareCrop(width: Int, height: Int): Boolean {
        val side = minOf(width, height).toFloat()
        val marginX = (1 - side / width) / 2
        val marginY = (1 - side / height) / 2
        return centerX in marginX..(1 - marginX) && centerY in marginY..(1 - marginY)
    }

    companion object {
        fun fromPixels(left: Int, top: Int, right: Int, bottom: Int, width: Int, height: Int) =
            NormalizedBox(
                left = (left.toFloat() / width).coerceIn(0f, 1f),
                top = (top.toFloat() / height).coerceIn(0f, 1f),
                right = (right.toFloat() / width).coerceIn(0f, 1f),
                bottom = (bottom.toFloat() / height).coerceIn(0f, 1f),
            )
    }
}

/** A recognised note and where it is in the frame. */
data class NoteHighlight(val label: String, val box: NormalizedBox)
