"""Create the two topics (idempotent). Also imported by the other scripts."""
import time

from kafka import KafkaAdminClient
from kafka.admin import NewTopic
from kafka.errors import TopicAlreadyExistsError

import settings


def ensure_topics(retries=10):
    admin = None
    last_error = None
    for attempt in range(retries):
        try:
            admin = KafkaAdminClient(bootstrap_servers=settings.BOOTSTRAP)
            break
        except Exception as exc:  # exception class names for "no broker" differ between kafka-python versions
            last_error = exc
            print("Kafka not reachable yet at %s (attempt %d/%d): %s"
                  % (settings.BOOTSTRAP, attempt + 1, retries, type(exc).__name__))
            time.sleep(3)
    if admin is None:
        raise SystemExit("Could not reach Kafka at %s (%r). Is Docker Desktop running and "
                         "`docker compose up -d` done?" % (settings.BOOTSTRAP, last_error))

    # ONE partition on purpose: the GNN needs every sensor of a time step together and in
    # order, so a single ordered log per topic is the right choice for this snapshot stream.
    for name in (settings.TOPIC_RAW, settings.TOPIC_PRED):
        try:
            admin.create_topics([NewTopic(name=name, num_partitions=1, replication_factor=1)])
            print("created topic", name)
        except TopicAlreadyExistsError:
            print("topic exists:", name)
    admin.close()


if __name__ == "__main__":
    ensure_topics()