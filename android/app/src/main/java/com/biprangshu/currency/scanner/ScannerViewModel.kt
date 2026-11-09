package com.biprangshu.currency.scanner

import android.app.Application
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.lifecycle.AndroidViewModel
import com.biprangshu.currency.R
import com.biprangshu.currency.recognition.CurrencyClassifier
import com.biprangshu.currency.recognition.Stabilizer
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import java.io.IOException
import java.util.concurrent.Executor
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class ScannerViewModel(private val app: Application) : AndroidViewModel(app) {

    private val _state = MutableStateFlow(ScannerState())
    val state = _state.asStateFlow()

    private val announcer = Announcer(app) { showError(ScannerError.Speech) }

    // Everything below is confined to this single thread, including closing the native interpreter.
    private val analysisThread: ExecutorService = Executors.newSingleThreadExecutor()
    private val stabilizer = Stabilizer()
    private var classifier: CurrencyClassifier? = null

    val analysisExecutor: Executor get() = analysisThread
    val analyzer = ImageAnalysis.Analyzer { image -> image.use(::analyze) }

    init {
        analysisThread.execute(::loadClassifier)
    }

    fun onAction(action: ScannerAction) {
        when (action) {
            ScannerAction.OnRepeatClick -> repeatLast()
            ScannerAction.OnTorchClick -> _state.update { it.copy(isTorchOn = !it.isTorchOn) }
            is ScannerAction.OnCameraReady -> _state.update { it.copy(hasTorch = action.hasTorch) }
            ScannerAction.OnCameraFailed -> showError(ScannerError.Camera)
            ScannerAction.OnScreenStopped -> onScreenStopped()
        }
    }

    private fun loadClassifier() {
        try {
            classifier = CurrencyClassifier.load(app)
        } catch (e: IOException) {
            onLoadFailed(e)
        } catch (e: IllegalArgumentException) {
            onLoadFailed(e)
        }
    }

    private fun onLoadFailed(cause: Exception) {
        Log.e(TAG, "Could not load the model", cause)
        showError(ScannerError.Model)
    }

    private fun analyze(image: ImageProxy) {
        val classifier = classifier ?: return
        val plane = image.planes[0]
        val prediction = classifier.classify(plane.buffer, image.width, image.height, plane.rowStride)
        val denomination = stabilizer.update(prediction.label, prediction.confidence) ?: return
        _state.update { it.copy(lastDenomination = denomination) }
        announcer.announceDetection(spokenDenomination(denomination))
    }

    private fun repeatLast() {
        val denomination = _state.value.lastDenomination
        announcer.speak(
            if (denomination != null) spokenDenomination(denomination)
            else app.getString(R.string.spoken_no_note),
        )
    }

    private fun onScreenStopped() {
        // The camera turns the torch off when it closes, and a note held up after returning to
        // the app should be announced even if it was the last one spoken.
        _state.update { it.copy(isTorchOn = false) }
        analysisThread.execute(stabilizer::reset)
        announcer.stop()
    }

    private fun spokenDenomination(denomination: String) =
        app.getString(R.string.spoken_denomination, denomination)

    private fun showError(error: ScannerError) {
        _state.update { it.copy(error = error) }
    }

    override fun onCleared() {
        analysisThread.execute {
            classifier?.close()
            classifier = null
        }
        analysisThread.shutdown()
        announcer.shutdown()
    }

    private companion object {
        const val TAG = "ScannerViewModel"
    }
}
