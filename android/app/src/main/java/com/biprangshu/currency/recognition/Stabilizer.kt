package com.biprangshu.currency.recognition

/**
 * Decides when a stream of per-frame predictions should be spoken.
 *
 * A note is reported once the same non-background label has won with confidence of at least
 * [threshold] in [framesRequired] of the last [windowSize] frames, so one blurry frame does not
 * restart the wait. It is not reported again until background has been seen for [framesRequired]
 * frames in a row, so a misclassified frame while a note is held still does not make the app
 * repeat itself. A different note is reported at once.
 */
class Stabilizer(
    private val threshold: Float = 0.70f,
    private val framesRequired: Int = 4,
    private val windowSize: Int = 6,
    private val backgroundLabel: String = "background",
) {
    // One entry per recent frame: the label if it qualified, otherwise null.
    private val window = ArrayDeque<String?>(windowSize + 1)
    private var backgroundFrames = 0
    private var lastReported: String? = null

    /** @return the label to announce, or null if nothing should be said for this frame. */
    fun update(label: String, confidence: Float): String? {
        if (label == backgroundLabel) {
            remember(null)
            backgroundFrames++
            if (backgroundFrames >= framesRequired) lastReported = null
            return null
        }

        backgroundFrames = 0
        remember(label.takeIf { confidence >= threshold })
        if (window.count { it == label } < framesRequired || label == lastReported) return null

        lastReported = label
        return label
    }

    fun reset() {
        window.clear()
        backgroundFrames = 0
        lastReported = null
    }

    private fun remember(entry: String?) {
        window.addLast(entry)
        if (window.size > windowSize) window.removeFirst()
    }
}
