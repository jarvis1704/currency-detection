package com.biprangshu.currency.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable

// Fixed rather than dynamic: low-vision users need maximum contrast regardless of wallpaper.
private val HighContrastColorScheme = darkColorScheme(
    primary = SignalYellow,
    onPrimary = PureBlack,
    background = PureBlack,
    onBackground = PureWhite,
    surface = PureBlack,
    onSurface = PureWhite,
    onSurfaceVariant = LightGrey,
    error = ErrorPink,
    onError = PureBlack,
)

@Composable
fun CurrencyReaderTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = HighContrastColorScheme,
        typography = Typography,
        content = content,
    )
}
