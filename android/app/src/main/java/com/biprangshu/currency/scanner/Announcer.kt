package com.biprangshu.currency.scanner

import android.content.Context
import android.os.VibrationEffect
import android.os.Vibrator
import android.speech.tts.TextToSpeech
import java.util.Locale

/** Speaks and vibrates. Safe to call from any thread. */
class Announcer(
    context: Context,
    private val onSpeechUnavailable: () -> Unit,
) : TextToSpeech.OnInitListener {

    private val vibrator = context.getSystemService(Vibrator::class.java)
    private val lock = Any()
    private var ready = false
    private var pending: String? = null
    private val tts = TextToSpeech(context.applicationContext, this)

    override fun onInit(status: Int) {
        val languageSet = status == TextToSpeech.SUCCESS &&
            (setLanguage(Locale.forLanguageTag("en-IN")) || setLanguage(Locale.ENGLISH))
        if (!languageSet) {
            onSpeechUnavailable()
            return
        }
        val queued = synchronized(lock) {
            ready = true
            pending.also { pending = null }
        }
        queued?.let(::speakNow)
    }

    fun speak(text: String) {
        val canSpeak = synchronized(lock) {
            if (!ready) pending = text
            ready
        }
        if (canSpeak) speakNow(text)
    }

    fun announceDetection(text: String) {
        vibrator.vibrate(VibrationEffect.createOneShot(VIBRATION_MS, VibrationEffect.DEFAULT_AMPLITUDE))
        speak(text)
    }

    fun stop() {
        synchronized(lock) { pending = null }
        tts.stop()
    }

    fun shutdown() = tts.shutdown()

    private fun setLanguage(locale: Locale): Boolean =
        tts.setLanguage(locale) >= TextToSpeech.LANG_AVAILABLE

    private fun speakNow(text: String) {
        tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, UTTERANCE_ID)
    }

    private companion object {
        const val VIBRATION_MS = 150L
        const val UTTERANCE_ID = "announcement"
    }
}
