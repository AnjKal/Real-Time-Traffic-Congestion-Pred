Agent 3 — User Interaction Agent
This guide supplements the main project README. The Interaction Agent uses Amazon Bedrock Converse and existing traffic, incident and journey data to answer questions, maintain conversation history and generate subscription-based notifications. It does not train DCRNN or calculate new routes independently.


1. Prerequisites
- Git, Docker Desktop with Compose, and a TomTom API key for live collection.
- AWS account with Bedrock model access and AWS CLI v2 for Bedrock mode.
- Python 3.11 recommended for local webapp development. Docker handles the separate pipeline Python environment.


2. Get the code
git clone --branch agentic https://github.com/AnjKal/Real-Time-Traffic-Congestion-Pred.git
cd Real-Time-Traffic-Congestion-Pred
For an existing clone, switch to agentic and pull from the team repository (use team or origin according to git remote -v). Preserve local changes before switching or pulling.
git switch agentic
git pull team agentic


3. Configure live traffic
If .env does not already exist, run:
Copy-Item .env.example .env
Set TOMTOM_API_KEY in .env to your own key. Keep .env private. The pipeline uses this key to collect traffic; the webapp reads the published data. Follow the main README for any required model/data assets.


4. Run with Docker (mock mode)
docker compose up -d --build
docker compose ps -a
Invoke-RestMethod http://localhost:8001/health
Wait until the model service reports ready: true. Open http://localhost:8010. Kafka, topic initialization, pipeline and webapp are managed by Compose. The default mock provider is deterministic and does not require AWS; it is not a general-purpose LLM. Live collection still requires TomTom access. Initial builds may take considerable time.


5. Enable Amazon Bedrock (real LLM)
Use your own AWS account and credentials. The tested model is Amazon Nova Lite (amazon.nova-lite-v1:0) in us-east-1. Verify model availability and account permissions. Prefer temporary credentials where practical. For local development, an IAM identity with the following narrowly scoped policy can invoke the tested model directly:
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":"bedrock:InvokeModel","Resource":"arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-lite-v1:0"}]}
Configure your own AWS CLI profile (never use another teammate's keys):
aws configure --profile traffic-agent
aws sts get-caller-identity --profile traffic-agent
Use us-east-1 as the region for the tested configuration. If using temporary credentials or SSO, configure the corresponding credential provider; the static-profile Docker example below may need adaptation.

Create docker-compose.local.yml in the project root (this file is ignored by Git):
services:
  webapp:
    environment:
      INTERACTION_LLM_PROVIDER: bedrock
      AWS_PROFILE: traffic-agent
      AWS_REGION: us-east-1
      BEDROCK_MODEL_ID: amazon.nova-lite-v1:0
      AWS_SHARED_CREDENTIALS_FILE: /root/.aws/credentials
      AWS_CONFIG_FILE: /root/.aws/config
    volumes:
      - "${USERPROFILE}/.aws/credentials:/root/.aws/credentials:ro"
      - "${USERPROFILE}/.aws/config:/root/.aws/config:ro"
This Windows local-development example mounts AWS profile files read-only into the webapp container. It exposes other profiles in those files to that container as well; use dedicated credential files for better isolation. Never bake credentials into images or commit them.
docker compose -f docker-compose.yml -f docker-compose.local.yml config --quiet
docker compose -f docker-compose.yml -f docker-compose.local.yml up -d --build
Invoke-RestMethod http://localhost:8010/api/interaction/status | ConvertTo-Json -Depth 5
Confirm provider bedrock, then send a chat message to verify actual inference. Status alone does not prove AWS invocation succeeds. Bedrock and TomTom usage may incur charges.


6. Use the dashboard
Open http://localhost:8010. Open the chat panel and ask about current congestion, incidents or system status. To test journey questions, choose a source and destination, generate routes, select a route and start a journey. Journey movement is simulated; traffic collection is live when configured. Forecasts in pipeline-test mode are not validated accuracy claims.

Example questions: Which areas have heavy traffic? How much longer until I arrive? Am I running late? Are there incidents ahead? Would switching routes help?

Use the journey monitoring/subscription controls to receive notifications. The notification engine polls every 120 seconds by default and uses explicit subscriptions; a short simulated journey may finish before a polling cycle. Conversation history and notification state are stored in the ignored data/live/interaction_memory.sqlite database.


7. Architecture and API
The webapp Interaction Agent reads existing published incident, feature and journey snapshots and model health. Its six read-only tools are get_incidents, get_congestion_summary, get_journeys, get_journey, get_route_recommendations, and get_system_status. Bedrock selects tools and generates grounded answers; validation and fallback logic reduce unsupported claims. Notifications are generated independently of the LLM.

Key endpoints:
Method	Endpoint
POST	/api/chat
POST	/api/interaction/session
GET/DELETE	/api/chat/history
GET	/api/notifications
GET	/api/notifications/unread-count
POST	/api/notifications/read
GET/POST	/api/subscriptions
DELETE	/api/subscriptions/{journey_id}
GET	/api/interaction/status


Session-scoped operations require the session ID and bearer token as defined by the API. Do not share tokens. The browser manages these for normal dashboard use.

Backend checks:
Invoke-RestMethod http://localhost:8010/api/journeys | ConvertTo-Json -Depth 10
Invoke-RestMethod http://localhost:8010/api/incidents | ConvertTo-Json -Depth 10
Invoke-RestMethod http://localhost:8010/api/switches | ConvertTo-Json -Depth 10

8. Local development and tests
From the project root, using a suitable Python environment:
python -m pip install -r requirements_webapp.txt
python -m pip install pytest
python -m unittest discover -s . -p "test_interaction_agent*.py" -q
python -m pytest test_webapp_offline.py test_stage2_features.py -q
The final handover recorded 355 passing Interaction Agent unittest tests; Copilot reported 375 passing tests in its combined regression run. The latest live smoke test still found response relevance issues. Intentional simulated exception logs in tests are expected when the suite ends with OK.

For offline development, run python dev_stub_model_service.py in one terminal and python run_webapp.py --port 8010 in another. Stop conflicting Docker services first. This does not provide a live traffic pipeline.

9. Maintenance and troubleshooting
- Model not ready: inspect docker compose logs -f pipeline and wait for initialization.
- Bedrock errors: check AWS profile, region, permissions, model access and credentials inside the container. Check docker compose logs --tail 100 webapp for safe grounding diagnostics.
- Missing/stale data: check pipeline health, TomTom quota and published snapshots. Do not treat stale data as current.
- After editing the TomTom key in .env, recreate the pipeline without rebuilding:
docker compose -f docker-compose.yml -f docker-compose.local.yml up -d --no-build --force-recreate --no-deps pipeline
- After changing webapp code or dependencies, rebuild the webapp:
docker compose -f docker-compose.yml -f docker-compose.local.yml up -d --build --no-deps webapp
Use only the base Compose file in these commands if running without the local override. Stop services when not testing to avoid unnecessary API usage:
docker compose stop


10. Handover and known limitations
Core implementation, Bedrock invocation, live data integration and automated tests are in place. Remaining work: improve semantic tool selection and response relevance, investigate irrelevant grounding fallbacks and incomplete requests, verify numerical claims against contemporaneous snapshots, and complete live notification delivery/deduplication testing. The model can still produce imperfect answers; passing automated tests does not establish complete live reliability. Forecast horizons are limited to the published data (currently 2–24 minutes in the tested configuration). No guarantee of arrival time or road safety is provided.

This is a development setup, not a production security deployment. Session tokens have no expiry/rotation in the current implementation; production authentication, rate limiting and HTTPS require further work. Keep .env, docker-compose.local.yml, AWS credentials, runtime databases and temporary reports out of commits.