from datetime import datetime
import cv2
import numpy as np
from confluent_kafka import Consumer


def read_config():
    # reads the client configuration from client.properties
    # and returns it as a key-value map
    config = {}
    with open("client.properties") as fh:
        for line in fh:
            line = line.strip()
            if len(line) != 0 and line[0] != "#":
                parameter, value = line.strip().split("=", 1)
                config[parameter] = value.strip()
    return config


def consume(topic, consumer):
    msg = consumer.poll(1.0)
    if msg is not None and msg.error() is None:
        _date = datetime.fromtimestamp(msg.timestamp()[1] / 1e3)
        date = _date.strftime("%Y/%m/%d %H:%M:%S.%f")
        jpeg_frame = np.frombuffer(msg.value(), np.uint8)
        print(f"{topic}:{date} = {jpeg_frame}")
        frame = cv2.imdecode(jpeg_frame, cv2.IMREAD_COLOR)
        return frame
    else:
        print("No new messages")
        return None


def get_consumer(topic, group_id="python-group-1", offset="latest"):
    # sets the consumer group ID and offset
    config = read_config()
    config["group.id"] = group_id
    config["auto.offset.reset"] = offset

    # creates a new consumer instance
    consumer = Consumer(config)

    # subscribes to the specified topic
    consumer.subscribe([topic])
    return consumer


def main():
    topic = "camera_07"
    consumer = get_consumer(topic)

    try:
        while True:
            # consumer polls the topic and prints any incoming messages
            frame = consume(topic, consumer)

    except KeyboardInterrupt:
        pass
    finally:
        # closes the consumer connection
        consumer.close()


if __name__ == "__main__":
    main()
