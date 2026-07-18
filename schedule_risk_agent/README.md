# Schedule Risk Agent

## Development commands

Train the development three-bin model:

    python3 -m schedule_risk_agent.train

Build and publish a local feature snapshot from Snowflake:

    python3 -m schedule_risk_agent.feature_refresh --target local

Run tests:

    python3 -m pytest -q

Run the MCP service in the Python 3.11 container:

    docker compose -f docker-compose.schedule-risk.yml up schedule-risk-agent

The local publisher and repository are interim storage implementations. The
production design publishes to and reads from the persistent Snowflake
SCHEDULE_PROJECT_FEATURES_CURRENT table.

