# Project Documentation

## Overview
This repository contains an Armenian legal chat application, a real-time vision analysis pipeline, and supporting documentation. It now supports live webcam and network camera streams for real-time Armenian action detection and emotion inference.

## Project Summary
- Added shared vision classifier support in `src/services/vision_classifier.py`.
- Added Armenian body-language and activity detection support in `src/services/vision.py`.
- Added emotion detection support using face landmarks and heuristic inference.
- Added live camera source selection in `src/main.py` for built-in webcams and IP/mobile streams.
- Documented usage for `0`, `http://...`, and `rtsp://...` camera sources.
- Maintained the configured GitHub remote `https://github.com/Tsovinar1986/armenian_legal_chat.git`.

## Implemented updates
- Real-time webcam and network camera streaming support.
- Armenian-language action detection for gestures like hands-on-hips, pointing, walking, running, and more.
- Emotion detection support for facial expression inference.
- Documentation updates to mirror the new live stream and vision capabilities.

## Recent Changes
- Added camera source prompt and IP stream fallbacks in `src/main.py`.
- Extended action detection heuristics in `src/services/vision.py`.
- Updated `README.md`, `DOCUMENTATION.md`, and `DOCUMENTATION_hy.md` with the new runtime instructions.

## Repository Notes
- Target remote: https://github.com/Tsovinar1986/armenian_legal_chat.git
- Commit and push documentation changes after verifying the working tree.

## Next steps
- Test the live camera stream with both built-in webcam and IP/mobile sources.
- Enhance object detection and scene understanding support.
- Continue refining Armenian vision feedback and documentation.
