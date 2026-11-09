package com.biprangshu.currency

import android.media.AudioManager
import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.biprangshu.currency.scanner.ScannerRoot
import com.biprangshu.currency.ui.theme.CurrencyReaderTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        // Scanning involves no touches, so the screen would otherwise time out mid-use.
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        // Volume keys should control the announcements, which play on the media stream.
        volumeControlStream = AudioManager.STREAM_MUSIC
        setContent {
            CurrencyReaderTheme {
                ScannerRoot()
            }
        }
    }
}
