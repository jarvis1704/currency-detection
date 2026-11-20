package com.biprangshu.currency.recognition

/**
 * Decides which note, if any, should be outlined in the viewfinder for each frame.
 *
 * A note is outlined on frames where the classifier is confident it is not background and the
 * locator found where it is. The outline survives up to [holdFrames] frames without a fresh box,
 * so one blurry frame or one missed detection does not make it flicker.
 */
class NoteHighlighter(
    private val threshold: Float = 0.70f,
    private val holdFrames: Int = 3,
    private val backgroundLabel: String = "background",
) {
    private var current: NoteHighlight? = null
    private var misses = 0

    /** @param locate finds the note in the frame; only called when the frame shows a note. */
    fun update(label: String, confidence: Float, locate: () -> NormalizedBox?): NoteHighlight? {
        val isNote = label != backgroundLabel && confidence >= threshold
        val box = if (isNote) locate() else null
        if (box != null) {
            current = NoteHighlight(label, box)
            misses = 0
        } else if (current != null && ++misses > holdFrames) {
            current = null
            misses = 0
        }
        return current
    }

    fun reset() {
        current = null
        misses = 0
    }
}
