package com.biprangshu.currency.recognition

import android.content.Context
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

data class Prediction(val label: String, val confidence: Float)

/** Not thread-safe: call [classify] and [close] from one thread. */
class CurrencyClassifier private constructor(
    private val interpreter: Interpreter,
    private val labels: List<String>,
) : AutoCloseable {

    private val input = ByteBuffer
        .allocateDirect(MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * MODEL_INPUT_CHANNELS * Float.SIZE_BYTES)
        .order(ByteOrder.nativeOrder())
    private val output = Array(1) { FloatArray(labels.size) }

    fun classify(rgba: ByteBuffer, width: Int, height: Int, rowStride: Int): Prediction {
        writeModelInput(rgba, width, height, rowStride, input)
        interpreter.run(input, output)
        val scores = output[0]
        val best = scores.indices.maxBy { scores[it] }
        return Prediction(labels[best], scores[best])
    }

    override fun close() = interpreter.close()

    companion object {
        private const val MODEL_ASSET = "currency_model.tflite"
        private const val LABELS_ASSET = "labels.txt"
        private const val THREADS = 2

        /** @throws IllegalArgumentException if the assets break the training hand-off contract. */
        fun load(context: Context): CurrencyClassifier {
            val labels = context.assets.open(LABELS_ASSET).bufferedReader().useLines { lines ->
                lines.map(String::trim).filter(String::isNotEmpty).toList()
            }
            val options = Interpreter.Options().setNumThreads(THREADS)
            val interpreter = Interpreter(mapModel(context), options)
            try {
                validate(interpreter, labels)
            } catch (e: IllegalArgumentException) {
                interpreter.close()
                throw e
            }
            return CurrencyClassifier(interpreter, labels)
        }

        private fun mapModel(context: Context): MappedByteBuffer =
            context.assets.openFd(MODEL_ASSET).use { descriptor ->
                FileInputStream(descriptor.fileDescriptor).use { stream ->
                    stream.channel.map(
                        FileChannel.MapMode.READ_ONLY,
                        descriptor.startOffset,
                        descriptor.declaredLength,
                    )
                }
            }

        private fun validate(interpreter: Interpreter, labels: List<String>) {
            val input = interpreter.getInputTensor(0)
            val output = interpreter.getOutputTensor(0)
            val expectedInput = listOf(1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, MODEL_INPUT_CHANNELS)
            require(input.shape().toList() == expectedInput && input.dataType() == DataType.FLOAT32) {
                "Unexpected model input: ${input.shape().toList()} ${input.dataType()}"
            }
            require(output.shape().toList() == listOf(1, labels.size) && output.dataType() == DataType.FLOAT32) {
                "Model output ${output.shape().toList()} ${output.dataType()} does not match ${labels.size} labels"
            }
        }
    }
}
