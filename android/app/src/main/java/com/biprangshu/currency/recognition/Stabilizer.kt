package com.biprangshu.currency.recognition

/**
 * Decides when a stream of per-frame predictions should be spoken.
 *
 * A note is reported once the same non-background label wins with confidence of at least
 * [threshold] for [framesRequired] frames in a row. It is not reported again until background has
 * been seen, also for [framesRequired] frames in a row, so that a single misclassified frame while
 * a note is held still does not make the app repeat itself. A different note is reported at once.
 */
class Stabilizer(
    private val threshold: Float = 0.70f,
    private val framesRequired: Int = 5,
    private val backgroundLabel: String = "background",
) {
    private var candidate: String? = null
    private var candidateFrames = 0
    private var backgroundFrames = 0
    private var lastReported: String? = null

    /** @return the label to announce, or null if nothing should be said for this frame. */
    fun update(label: String, confidence: Float): String? {
        if (label == backgroundLabel) {
            clearCandidate()
            backgroundFrames++
            if (backgroundFrames >= framesRequired) lastReported = null
            return null
        }

        backgroundFrames = 0
        if (confidence < threshold) {
            clearCandidate()
            return null
        }
        if (label != candidate) {
            candidate = label
            candidateFrames = 0
        }
        candidateFrames++
        if (candidateFrames < framesRequired || label == lastReported) return null

        lastReported = label
        return label
    }

    fun reset() {
        clearCandidate()
        backgroundFrames = 0
        lastReported = null
    }

    private fun clearCandidate() {
        candidate = null
        candidateFrames = 0
    }
}
