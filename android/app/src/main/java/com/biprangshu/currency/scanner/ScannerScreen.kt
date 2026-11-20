package com.biprangshu.currency.scanner

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.provider.Settings
import android.util.Log
import android.util.Size
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.compose.CameraXViewfinder
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview as CameraPreview
import androidx.camera.core.SurfaceRequest
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.lifecycle.awaitInstance
import androidx.camera.viewfinder.compose.MutableCoordinateTransformer
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.core.net.toUri
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.LifecycleEventEffect
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.biprangshu.currency.R
import com.biprangshu.currency.ui.theme.CurrencyReaderTheme
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.awaitCancellation

private const val TAG = "ScannerScreen"

// Larger frames only cost time: the model sees a 224×224 crop.
private val ANALYSIS_SIZE = Size(640, 480)

// Preview and analysis must share an aspect ratio, and so the same field of view, for boxes found
// on analysis frames to line up with the preview (see NoteHighlightOverlay).
private val SHARED_ASPECT_RATIO = AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY

@Composable
fun ScannerRoot(viewModel: ScannerViewModel = viewModel()) {
    val context = LocalContext.current
    var hasPermission by remember { mutableStateOf(context.hasCameraPermission()) }
    var permissionRequested by rememberSaveable { mutableStateOf(false) }
    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted -> hasPermission = granted }

    LaunchedEffect(Unit) {
        if (!hasPermission && !permissionRequested) {
            permissionRequested = true
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }
    // The user may have granted the permission in system settings while away.
    LifecycleEventEffect(Lifecycle.Event.ON_RESUME) {
        hasPermission = context.hasCameraPermission()
    }

    if (hasPermission) {
        ScannerCamera(viewModel)
    } else {
        CameraPermissionScreen(onOpenSettings = context::openAppSettings)
    }
}

@Composable
private fun ScannerCamera(viewModel: ScannerViewModel) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    var surfaceRequest by remember { mutableStateOf<SurfaceRequest?>(null) }
    var camera by remember { mutableStateOf<Camera?>(null) }
    val preview = remember {
        CameraPreview.Builder()
            .setResolutionSelector(
                ResolutionSelector.Builder().setAspectRatioStrategy(SHARED_ASPECT_RATIO).build(),
            )
            .build()
    }
    val analysis = remember {
        ImageAnalysis.Builder()
            .setResolutionSelector(
                ResolutionSelector.Builder()
                    .setAspectRatioStrategy(SHARED_ASPECT_RATIO)
                    .setResolutionStrategy(
                        ResolutionStrategy(
                            ANALYSIS_SIZE,
                            ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER,
                        ),
                    )
                    .build(),
            )
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
    }

    LaunchedEffect(lifecycleOwner) {
        preview.setSurfaceProvider { surfaceRequest = it }
        analysis.setAnalyzer(viewModel.analysisExecutor, viewModel.analyzer)
        val provider = ProcessCameraProvider.awaitInstance(context)
        try {
            val bound = try {
                provider.bindToLifecycle(
                    lifecycleOwner,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    analysis,
                )
            } catch (e: CancellationException) {
                throw e
            } catch (e: RuntimeException) {
                Log.e(TAG, "Could not bind the camera", e)
                viewModel.onAction(ScannerAction.OnCameraFailed)
                return@LaunchedEffect
            }
            camera = bound
            viewModel.onAction(ScannerAction.OnCameraReady(bound.cameraInfo.hasFlashUnit()))
            awaitCancellation()
        } finally {
            provider.unbind(preview, analysis)
            analysis.clearAnalyzer()
        }
    }

    LaunchedEffect(camera, state.isTorchOn) {
        camera?.cameraControl?.enableTorch(state.isTorchOn)
    }

    LifecycleEventEffect(Lifecycle.Event.ON_STOP) {
        viewModel.onAction(ScannerAction.OnScreenStopped)
    }

    ScannerScreen(
        state = state,
        surfaceRequest = surfaceRequest,
        onAction = viewModel::onAction,
    )
}

@Composable
fun ScannerScreen(
    state: ScannerState,
    surfaceRequest: SurfaceRequest?,
    onAction: (ScannerAction) -> Unit,
    modifier: Modifier = Modifier,
) {
    val haptic = LocalHapticFeedback.current
    val scannerDescription = state.lastDenomination
        ?.let { stringResource(R.string.a11y_scanner_last, it) }
        ?: stringResource(R.string.a11y_scanner_ready)
    val repeatLabel = stringResource(R.string.a11y_repeat_action)
    val coordinateTransformer = remember { MutableCoordinateTransformer() }

    Column(
        modifier = modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.surface)
            .semantics { contentDescription = scannerDescription }
            .clickable(onClickLabel = repeatLabel) {
                haptic.performHapticFeedback(HapticFeedbackType.KeyboardTap)
                onAction(ScannerAction.OnRepeatClick)
            },
    ) {
        Box(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
                .clipToBounds(),
        ) {
            surfaceRequest?.let { request ->
                CameraXViewfinder(
                    surfaceRequest = request,
                    coordinateTransformer = coordinateTransformer,
                    modifier = Modifier.fillMaxSize(),
                )
                NoteHighlightOverlay(
                    highlight = state.highlight,
                    surfaceResolution = request.resolution,
                    coordinateTransformer = coordinateTransformer,
                )
            }
        }

        Column(
            modifier = Modifier
                .fillMaxWidth()
                .verticalScroll(rememberScrollState())
                .navigationBarsPadding()
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            ScanResult(denomination = state.lastDenomination)
            state.error?.let { error ->
                Text(
                    text = stringResource(error.message),
                    style = MaterialTheme.typography.bodyLarge,
                    color = MaterialTheme.colorScheme.error,
                    textAlign = TextAlign.Center,
                    // Its own merge boundary, or the parent's description would swallow it.
                    modifier = Modifier.semantics(mergeDescendants = true) {
                        liveRegion = LiveRegionMode.Assertive
                    },
                )
            }
            if (state.hasTorch) {
                Button(
                    onClick = {
                        haptic.performHapticFeedback(
                            if (state.isTorchOn) HapticFeedbackType.ToggleOff
                            else HapticFeedbackType.ToggleOn,
                        )
                        onAction(ScannerAction.OnTorchClick)
                    },
                    shape = RoundedCornerShape(28.dp),
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(72.dp),
                ) {
                    Text(
                        text = stringResource(
                            if (state.isTorchOn) R.string.torch_turn_off else R.string.torch_turn_on,
                        ),
                        style = MaterialTheme.typography.titleLarge,
                    )
                }
            }
        }
    }
}

/** Hidden from TalkBack: the note is already spoken and is in the scanner's description. */
@Composable
private fun ScanResult(denomination: String?) {
    if (denomination != null) {
        Text(
            text = stringResource(R.string.denomination_display, denomination),
            style = MaterialTheme.typography.displayLarge,
            color = MaterialTheme.colorScheme.primary,
            textAlign = TextAlign.Center,
            modifier = Modifier.clearAndSetSemantics {},
        )
    } else {
        Text(
            text = stringResource(R.string.hint_point_camera),
            style = MaterialTheme.typography.headlineMedium,
            color = MaterialTheme.colorScheme.onSurface,
            textAlign = TextAlign.Center,
            modifier = Modifier.clearAndSetSemantics {},
        )
    }
}

@Composable
private fun CameraPermissionScreen(onOpenSettings: () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.surface)
            .safeDrawingPadding()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(24.dp, Alignment.CenterVertically),
    ) {
        Text(
            text = stringResource(R.string.permission_rationale),
            style = MaterialTheme.typography.headlineMedium,
            color = MaterialTheme.colorScheme.onSurface,
            textAlign = TextAlign.Center,
        )
        Button(
            onClick = onOpenSettings,
            shape = RoundedCornerShape(28.dp),
            modifier = Modifier
                .fillMaxWidth()
                .height(72.dp),
        ) {
            Text(
                text = stringResource(R.string.permission_open_settings),
                style = MaterialTheme.typography.titleLarge,
            )
        }
    }
}

private fun Context.hasCameraPermission(): Boolean =
    ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
        PackageManager.PERMISSION_GRANTED

private fun Context.openAppSettings() {
    startActivity(
        Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, "package:$packageName".toUri()),
    )
}

@Preview
@Composable
private fun ScannerScreenPreview() {
    CurrencyReaderTheme {
        ScannerScreen(
            state = ScannerState(lastDenomination = "500", hasTorch = true),
            surfaceRequest = null,
            onAction = {},
        )
    }
}

@Preview
@Composable
private fun CameraPermissionScreenPreview() {
    CurrencyReaderTheme {
        CameraPermissionScreen(onOpenSettings = {})
    }
}
