# Live traffic pipeline

The live app now runs through Docker Compose. Follow the current setup, background startup,
topic inspection, and shutdown instructions in [README.md](README.md).

The pipeline polls at a two-minute cadence. TomTom readings enter `traffic.raw`; the raw consumer
assembles each poll and triggers inference; forecasts enter `traffic.predictions`; and the
predictions consumer supplies the HTTP API. The first forecast is produced after the first
consumed poll, with the model's missing historical steps padded and reported during warm-up.

The model was trained on 15-minute readings. The two-minute cadence is for pipeline testing, not
an accuracy claim.
