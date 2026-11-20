package com.biprangshu.currency.scanner

import androidx.annotation.StringRes
import com.biprangshu.currency.R
import com.biprangshu.currency.recognition.NoteHighlight

data class ScannerState(
    val lastDenomination: String? = null,
    /** The note currently in view, outlined in the viewfinder; null when none is. */
    val highlight: NoteHighlight? = null,
    val hasTorch: Boolean = false,
    val isTorchOn: Boolean = false,
    val error: ScannerError? = null,
)

enum class ScannerError(@param:StringRes val message: Int) {
    Model(R.string.error_model),
    Speech(R.string.error_speech),
    Camera(R.string.error_camera),
}

sealed interface ScannerAction {
    data object OnRepeatClick : ScannerAction
    data object OnTorchClick : ScannerAction
    data class OnCameraReady(val hasTorch: Boolean) : ScannerAction
    data object OnCameraFailed : ScannerAction
    data object OnScreenStopped : ScannerAction
}
