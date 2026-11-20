package com.biprangshu.currency.recognition

import android.util.Log
import androidx.camera.core.ImageProxy
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.objects.ObjectDetection
import com.google.mlkit.vision.objects.ObjectDetector
import com.google.mlkit.vision.objects.defaults.ObjectDetectorOptions
import java.util.concurrent.ExecutionException
import java.util.concurrent.TimeUnit
import java.util.concurrent.TimeoutException

/**
 * Finds where the note is in a frame. The currency model is a classifier and cannot say where the
 * note is, so this uses ML Kit's bundled, on-device object detector to find the most prominent
 * object near the centre of the frame. Nothing leaves the phone.
 *
 * Blocks the calling thread until the detector answers, so call it from the analysis thread only.
 * Not thread-safe: call [locate] and [close] from one thread.
 */
class NoteLocator : AutoCloseable {

    private val detector: ObjectDetector = ObjectDetection.getClient(
        ObjectDetectorOptions.Builder()
            .setDetectorMode(ObjectDetectorOptions.STREAM_MODE)
            .build(),
    )

    /** @return the note's box in the frame's own orientation, or null if it could not be found. */
    fun locate(image: ImageProxy): NormalizedBox? {
        val bitmap = image.toBitmap()
        val rotation = image.imageInfo.rotationDegrees
        // Detection works best on an upright image; its boxes come back in upright coordinates.
        val input = InputImage.fromBitmap(bitmap, rotation)
        val objects = try {
            Tasks.await(detector.process(input), TIMEOUT_MS, TimeUnit.MILLISECONDS)
        } catch (e: ExecutionException) {
            Log.w(TAG, "Detection failed", e)
            return null
        } catch (e: TimeoutException) {
            Log.w(TAG, "Detection timed out", e)
            return null
        } catch (e: InterruptedException) {
            Thread.currentThread().interrupt()
            return null
        }

        val uprightWidth = if (rotation % 180 == 0) bitmap.width else bitmap.height
        val uprightHeight = if (rotation % 180 == 0) bitmap.height else bitmap.width
        val rect = objects.firstOrNull()?.boundingBox ?: return null
        val box = NormalizedBox
            .fromPixels(rect.left, rect.top, rect.right, rect.bottom, uprightWidth, uprightHeight)
            .uprightToBuffer(rotation)
        return box.takeIf { it.isCenteredInSquareCrop(bitmap.width, bitmap.height) }
    }

    override fun close() = detector.close()

    private companion object {
        const val TAG = "NoteLocator"
        const val TIMEOUT_MS = 500L
    }
}
