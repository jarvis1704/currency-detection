package com.biprangshu.currency.scanner

import android.util.Size
import androidx.camera.viewfinder.compose.CoordinateTransformer
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.VectorConverter
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.geometry.Size as ComposeSize
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Matrix
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.TextMeasurer
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.dp
import com.biprangshu.currency.R
import com.biprangshu.currency.recognition.NormalizedBox
import com.biprangshu.currency.recognition.NoteHighlight
import kotlinx.coroutines.launch

private const val FADE_MS = 180
private val MarkerStroke = 4.dp
private val MarkerShadowStroke = 8.dp
private val MarkerMinArm = 16.dp
private val MarkerMaxArm = 48.dp
private val LabelPaddingH = 12.dp
private val LabelPaddingV = 4.dp
private val LabelGap = 8.dp

/**
 * Draws corner markers around the note in [highlight], with its denomination above them.
 *
 * The box is measured on the analysis frame. Analysis and preview share the sensor orientation and
 * the 4:3 aspect ratio, so scaling the normalised box by [surfaceResolution] places it on the
 * preview surface, and [coordinateTransformer] (the viewfinder's view → surface transform) is
 * inverted to bring it into this composable's space. The overlay must be laid out exactly over
 * the viewfinder.
 *
 * Purely visual: the note is already spoken and described to TalkBack, so this adds no semantics.
 */
@Composable
fun NoteHighlightOverlay(
    highlight: NoteHighlight?,
    surfaceResolution: Size,
    coordinateTransformer: CoordinateTransformer,
    modifier: Modifier = Modifier,
) {
    val box = remember { Animatable(Rect.Zero, Rect.VectorConverter) }
    val alpha = remember { Animatable(0f) }
    // Kept after the note leaves so the label can fade out with the markers.
    var label by remember { mutableStateOf<String?>(null) }

    LaunchedEffect(highlight) {
        if (highlight == null) {
            alpha.animateTo(0f, tween(FADE_MS))
            return@LaunchedEffect
        }
        label = highlight.label
        val target = highlight.box.toRect()
        // Appear where the note is rather than sliding in from where the last one was.
        if (alpha.value == 0f) {
            box.snapTo(target)
        } else {
            // Restarted every frame; Animatable keeps its velocity, so this also smooths jitter.
            launch { box.animateTo(target, spring(stiffness = Spring.StiffnessMediumLow)) }
        }
        alpha.animateTo(1f, tween(FADE_MS))
    }

    val markerColor = MaterialTheme.colorScheme.primary
    val labelColor = MaterialTheme.colorScheme.onPrimary
    val labelStyle = MaterialTheme.typography.titleMedium.copy(
        color = labelColor,
        fontWeight = FontWeight.Bold,
    )
    val labelText = label?.let { stringResource(R.string.denomination_display, it) }
    val textMeasurer = rememberTextMeasurer()

    Canvas(modifier = modifier.fillMaxSize()) {
        val opacity = alpha.value
        if (opacity == 0f) return@Canvas

        val surfaceToView = Matrix(coordinateTransformer.transformMatrix.values.copyOf())
            .apply { invert() }
        val normalized = box.value
        val rect = surfaceToView.map(
            Rect(
                left = normalized.left * surfaceResolution.width,
                top = normalized.top * surfaceResolution.height,
                right = normalized.right * surfaceResolution.width,
                bottom = normalized.bottom * surfaceResolution.height,
            ),
        )

        drawCornerMarkers(rect, markerColor, opacity)
        if (labelText != null) {
            drawLabel(rect, labelText, textMeasurer, labelStyle, markerColor, opacity)
        }
    }
}

private fun NormalizedBox.toRect() = Rect(left, top, right, bottom)

private fun DrawScope.drawCornerMarkers(rect: Rect, color: Color, alpha: Float) {
    val arm = (minOf(rect.width, rect.height) * 0.25f)
        .coerceIn(MarkerMinArm.toPx(), MarkerMaxArm.toPx())
        .coerceAtMost(minOf(rect.width, rect.height) / 2)
    val path = Path().apply {
        fun corner(corner: Offset, dx: Float, dy: Float) {
            moveTo(corner.x + dx, corner.y)
            lineTo(corner.x, corner.y)
            lineTo(corner.x, corner.y + dy)
        }
        corner(rect.topLeft, arm, arm)
        corner(rect.topRight, -arm, arm)
        corner(rect.bottomRight, -arm, -arm)
        corner(rect.bottomLeft, arm, -arm)
    }
    // A dark halo keeps the markers visible on a bright note or background.
    drawPath(
        path = path,
        color = Color.Black,
        alpha = alpha * 0.35f,
        style = Stroke(MarkerShadowStroke.toPx(), cap = StrokeCap.Round, join = StrokeJoin.Round),
    )
    drawPath(
        path = path,
        color = color,
        alpha = alpha,
        style = Stroke(MarkerStroke.toPx(), cap = StrokeCap.Round, join = StrokeJoin.Round),
    )
}

private fun DrawScope.drawLabel(
    rect: Rect,
    text: String,
    textMeasurer: TextMeasurer,
    style: TextStyle,
    background: Color,
    alpha: Float,
) {
    val layout = textMeasurer.measure(text, style)
    val pill = ComposeSize(
        width = layout.size.width + LabelPaddingH.toPx() * 2,
        height = layout.size.height + LabelPaddingV.toPx() * 2,
    )
    val gap = LabelGap.toPx()
    // Above the note when there is room, otherwise just inside its top edge.
    val top = if (rect.top - gap - pill.height >= 0f) rect.top - gap - pill.height else rect.top + gap
    val left = rect.left.coerceIn(0f, (size.width - pill.width).coerceAtLeast(0f))
    drawRoundRect(
        color = background,
        topLeft = Offset(left, top),
        size = pill,
        cornerRadius = CornerRadius(pill.height / 2),
        alpha = alpha,
    )
    drawText(
        textLayoutResult = layout,
        topLeft = Offset(left + LabelPaddingH.toPx(), top + LabelPaddingV.toPx()),
        alpha = alpha,
    )
}
