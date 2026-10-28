# Kafka container walkthrough

This guide explains how to inspect and exercise the Kafka environment defined in
[`docker-compose.yml`](./docker-compose.yml). The compose file starts:

- Apache Kafka `3.7.0`
- One broker with node id `1`
- KRaft mode (Kafka manages its own metadata; there is no ZooKeeper container)
- A combined broker/controller process
- Host access at `localhost:9092`
- Persistent data in the Docker volume `kafka-data`

The commands below use Docker Compose and the Kafka command-line tools inside the
container. They are written for PowerShell, but the commands are also valid in a
Linux/macOS shell unless PowerShell-specific syntax is used.

## 1. Check the required tools

Install and start Docker Desktop, then verify that Docker and Compose are
available:

```powershell
docker --version
docker compose version
docker info
```

Run all commands from the repository root:

```powershell
Set-Location C:\Users\Suresh\Documents\DCRNN
```

## 2. Configure and start Kafka

Start Kafka in the background and recreate it if the compose definition changed:

```powershell
docker compose up -d kafka
docker compose ps
```

Follow the startup log. Kafka is ready when the log no longer shows startup
errors and the broker has started accepting connections:

```powershell
docker compose logs -f kafka
```

Press `Ctrl+C` to stop following logs; this does not stop the container.

Useful lifecycle commands:

```powershell
# Stop the container but keep the container and named volume
docker compose stop kafka

# Start an existing stopped container
docker compose start kafka

# Stop and remove the container, but keep named volumes
docker compose down

# Recreate the container without deleting Kafka data
docker compose up -d --force-recreate kafka
```

## 3. Inspect the container and its configuration

Check the container state, ports, environment, mounts, network, and resource
usage:

```powershell
docker ps --filter "name=kafka"
docker inspect kafka
docker port kafka
docker stats kafka --no-stream
docker volume inspect dcrnn_kafka-data
docker network ls
docker compose config
```

The exact generated volume name can differ if the project directory has been
renamed. Find it with:

```powershell
docker volume ls --filter "name=kafka-data"
```

Open a shell inside the container:

```powershell
docker exec -it kafka /bin/bash
```

The Kafka CLI scripts are under `/opt/kafka/bin`. The commands in the rest of
this guide use `docker exec` so they can be copied without opening an
interactive shell.

## 4. Check broker, listener, and KRaft metadata

The compose configuration exposes the `PLAINTEXT` listener on port `9092`.
Port `9093` is the controller listener and is intentionally internal to the
container network.

```powershell
# Broker API version and connectivity
docker exec kafka /opt/kafka/bin/kafka-broker-api-versions.sh `
  --bootstrap-server localhost:9092

# Broker configuration currently applied by the image
docker exec kafka /opt/kafka/bin/kafka-configs.sh `
  --bootstrap-server localhost:9092 `
  --entity-type brokers --entity-name 1 --describe

# KRaft metadata quorum status
docker exec kafka /opt/kafka/bin/kafka-metadata-quorum.sh `
  --bootstrap-server localhost:9092 describe --status

# KRaft metadata log directory information
docker exec kafka /opt/kafka/bin/kafka-log-dirs.sh `
  --bootstrap-server localhost:9092 --describe
```

To see the environment variables passed by Compose:

```powershell
docker inspect kafka --format '{{range .Config.Env}}{{println .}}{{end}}'
```

## 5. Create and inspect a topic

Create a topic named `traffic-events` with three partitions and one replica.
There is only one broker, so the replication factor must be `1`:

```powershell
docker exec kafka /opt/kafka/bin/kafka-topics.sh `
  --bootstrap-server localhost:9092 `
  --create `
  --topic traffic-events `
  --partitions 3 `
  --replication-factor 1
```

List all topics, describe the topic, and inspect its topic-level configuration:

```powershell
docker exec kafka /opt/kafka/bin/kafka-topics.sh `
  --bootstrap-server localhost:9092 --list

docker exec kafka /opt/kafka/bin/kafka-topics.sh `
  --bootstrap-server localhost:9092 `
  --describe --topic traffic-events

docker exec kafka /opt/kafka/bin/kafka-configs.sh `
  --bootstrap-server localhost:9092 `
  --entity-type topics --entity-name traffic-events --describe
```

Change a topic configuration and verify it:

```powershell
docker exec kafka /opt/kafka/bin/kafka-configs.sh `
  --bootstrap-server localhost:9092 `
  --entity-type topics `
  --entity-name traffic-events `
  --alter --add-config retention.ms=86400000

docker exec kafka /opt/kafka/bin/kafka-configs.sh `
  --bootstrap-server localhost:9092 `
  --entity-type topics --entity-name traffic-events --describe
```

Delete the topic when finished experimenting:

```powershell
docker exec kafka /opt/kafka/bin/kafka-topics.sh `
  --bootstrap-server localhost:9092 `
  --delete --topic traffic-events
```

## 6. Produce messages

Start an interactive producer. Each line entered is one Kafka record:

```powershell
docker exec -it kafka /opt/kafka/bin/kafka-console-producer.sh `
  --bootstrap-server localhost:9092 `
  --topic traffic-events
```

Try these lines, one at a time:

```text
sensor-1,2026-10-13T10:00:00Z,42.5
sensor-2,2026-10-13T10:00:05Z,43.1
sensor-3,2026-10-13T10:00:10Z,41.9
```

Press `Ctrl+C` to stop the producer. To send a known batch non-interactively:

```powershell
"message-one`nmessage-two`nmessage-three" |
  docker exec -i kafka /opt/kafka/bin/kafka-console-producer.sh `
    --bootstrap-server localhost:9092 `
    --topic traffic-events
```

Kafka chooses a partition using the record key when one is supplied. The
following producer reads `key:value` input and uses the key for partitioning:

```powershell
docker exec -it kafka /opt/kafka/bin/kafka-console-producer.sh `
  --bootstrap-server localhost:9092 `
  --topic traffic-events `
  --property parse.key=true `
  --property key.separator=:
```

## 7. Consume messages and observe partitions

Read all available records from every partition, then exit when the current
records have been consumed:

```powershell
docker exec -it kafka /opt/kafka/bin/kafka-console-consumer.sh `
  --bootstrap-server localhost:9092 `
  --topic traffic-events `
  --from-beginning `
  --timeout-ms 10000
```

Display the partition and offset for every record:

```powershell
docker exec -it kafka /opt/kafka/bin/kafka-console-consumer.sh `
  --bootstrap-server localhost:9092 `
  --topic traffic-events `
  --from-beginning `
  --property print.partition=true `
  --property print.offset=true `
  --timeout-ms 10000
```

Read one partition only. This is useful for seeing that each partition has its
own ordered log:

```powershell
docker exec -it kafka /opt/kafka/bin/kafka-console-consumer.sh `
  --bootstrap-server localhost:9092 `
  --topic traffic-events `
  --partition 0 `
  --offset 0 `
  --property print.partition=true `
  --property print.offset=true `
  --timeout-ms 10000
```

Follow new records continuously:

```powershell
docker exec -it kafka /opt/kafka/bin/kafka-console-consumer.sh `
  --bootstrap-server localhost:9092 `
  --topic traffic-events `
  --group traffic-debug `
  --property print.partition=true `
  --property print.offset=true
```

Leave this consumer running, then use a second terminal to produce messages.
Press `Ctrl+C` when finished.

## 8. Consumer groups and offsets

A consumer group divides partitions among its active consumers. With three
partitions, up to three consumers in one group can actively consume in parallel.

```powershell
# List groups
docker exec kafka /opt/kafka/bin/kafka-consumer-groups.sh `
  --bootstrap-server localhost:9092 --list

# Show members, assignments, current offsets, and lag
docker exec kafka /opt/kafka/bin/kafka-consumer-groups.sh `
  --bootstrap-server localhost:9092 `
  --group traffic-debug --describe

# Show group state and member assignments
docker exec kafka /opt/kafka/bin/kafka-consumer-groups.sh `
  --bootstrap-server localhost:9092 `
  --group traffic-debug --describe --state

docker exec kafka /opt/kafka/bin/kafka-consumer-groups.sh `
  --bootstrap-server localhost:9092 `
  --group traffic-debug --describe --members --verbose
```

Reset a group's offsets only when no consumers from that group are running.
The preview shows the proposed reset without changing anything:

```powershell
docker exec kafka /opt/kafka/bin/kafka-consumer-groups.sh `
  --bootstrap-server localhost:9092 `
  --group traffic-debug `
  --topic traffic-events `
  --reset-offsets --to-earliest

docker exec kafka /opt/kafka/bin/kafka-consumer-groups.sh `
  --bootstrap-server localhost:9092 `
  --group traffic-debug `
  --topic traffic-events `
  --reset-offsets --to-earliest --execute
```

Other useful reset modes include `--to-latest`, `--shift-by -10`, and
`--to-datetime 2026-10-13T10:00:00.000`.

## 9. Inspect offsets, records, and storage

Show the earliest and latest offset for every partition:

```powershell
docker exec kafka /opt/kafka/bin/kafka-run-class.sh `
  kafka.tools.GetOffsetShell `
  --broker-list localhost:9092 `
  --topic traffic-events `
  --time -2

docker exec kafka /opt/kafka/bin/kafka-run-class.sh `
  kafka.tools.GetOffsetShell `
  --broker-list localhost:9092 `
  --topic traffic-events `
  --time -1
```

Inspect the broker's log directory and segment files:

```powershell
docker exec kafka /opt/kafka/bin/kafka-log-dirs.sh `
  --bootstrap-server localhost:9092 --describe

docker exec kafka /bin/bash -c "find /var/lib/kafka/data -maxdepth 2 -type f | sort"
```

The data directory includes topic-partition folders such as
`traffic-events-0`, `traffic-events-1`, and `traffic-events-2`, as well as
Kafka's internal topics.

## 10. Internal Kafka topics

Kafka creates internal topics for consumer offsets and, when transactions are
used, transaction state. Inspect them like any other topic:

```powershell
docker exec kafka /opt/kafka/bin/kafka-topics.sh `
  --bootstrap-server localhost:9092 --list

docker exec kafka /opt/kafka/bin/kafka-topics.sh `
  --bootstrap-server localhost:9092 `
  --describe --topic __consumer_offsets

docker exec kafka /opt/kafka/bin/kafka-topics.sh `
  --bootstrap-server localhost:9092 `
  --describe --topic __transaction_state
```

`__transaction_state` may not exist until transactional features are used.
Do not delete or alter internal topics during normal experiments.

## 11. Connectivity from the host and from another container

From the host, the advertised listener should resolve to:

```powershell
Test-NetConnection localhost -Port 9092
```

Applications running on the host should use:

```text
bootstrap.servers=localhost:9092
```

An application in another container on the same Compose network should use the
service/container name rather than `localhost`:

```text
bootstrap.servers=kafka:9092
```

Inspect the Compose network and verify name resolution from a temporary Kafka
image container:

```powershell
docker inspect kafka --format '{{json .NetworkSettings.Networks}}'
docker run --rm --network dcrnn_default apache/kafka:3.7.0 `
  /opt/kafka/bin/kafka-broker-api-versions.sh --bootstrap-server kafka:9092
```

If the Compose project name is not `dcrnn`, replace `dcrnn_default` with the
network shown by `docker inspect kafka`.

## 12. Reset the environment

To remove the container and all persisted Kafka data, use the named volume
explicitly. This deletes all topics and messages:

```powershell
docker compose down
docker volume ls --filter "name=kafka-data"
docker volume rm dcrnn_kafka-data
```

If the volume name differs, substitute the exact name returned by
`docker volume ls`. Start with a clean broker afterward:

```powershell
docker compose up -d kafka
```

## Quick learning sequence

For a short end-to-end exercise, run these steps in order:

1. Start Kafka with `docker compose up -d kafka`.
2. Create `traffic-events` with three partitions.
3. Start a consumer using `--group traffic-debug`.
4. Produce records with and without keys.
5. Inspect the consumer group with `--describe`.
6. Read records with partition and offset output enabled.
7. Compare earliest/latest offsets using `GetOffsetShell`.
8. Inspect broker logs, KRaft quorum status, and the Docker volume.
9. Stop the stack with `docker compose down` when finished.
