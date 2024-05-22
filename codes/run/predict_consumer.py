from datetime import datetime
import cv2
import numpy as np
from confluent_kafka import Consumer


def read_config():
    # reads the client configuration from client.properties
    # and returns it as a key-value map
    config = {}
    with open(
        "/home/jongphago/project/ultralytics/codes/kafka/client.properties"
    ) as fh:
        for line in fh:
            line = line.strip()
            if len(line) != 0 and line[0] != "#":
                parameter, value = line.strip().split("=", 1)
                config[parameter] = value.strip()
    return config


def consume(topic, config):
    # sets the consumer group ID and offset
    config["group.id"] = "python-group-1"
    config["auto.offset.reset"] = "latest"

    # creates a new consumer instance
    consumer = Consumer(config)

    # subscribes to the specified topic
    consumer.subscribe([topic])

    try:
        while True:
            # consumer polls the topic and prints any incoming messages
            msg = consumer.poll(1.0)
            if msg is not None and msg.error() is None:
                _date = datetime.fromtimestamp(msg.timestamp()[1] / 1e3)
                date = _date.strftime("%Y/%m/%d %H:%M:%S.%f")
                jpeg_frame = np.frombuffer(msg.value(), np.uint8)
                print(f"{topic}:{date} = {jpeg_frame}")
                frame = cv2.imdecode(jpeg_frame, cv2.IMREAD_COLOR)
                cv2.imshow("CCTV", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break

    except KeyboardInterrupt:
        pass
    finally:
        # closes the consumer connection
        consumer.close()


def main():
    config = read_config()
    topic = "camera_07"

    consume(topic, config)


if __name__ == "__main__":
    main()
