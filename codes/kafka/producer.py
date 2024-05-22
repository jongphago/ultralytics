from datetime import datetime
from http.client import CONTINUE
import cv2
from confluent_kafka import Producer, Consumer
from confluent_kafka import KafkaError


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


def produce(topic, producer, value, callback=None, verbose=True):
    # Produce a message to the topic
    producer.produce(
        topic,
        key=None,
        value=value,
        callback=callback,
    )
    if verbose:
        _date = datetime.now()
        date = _date.strftime("%Y/%m/%d %H:%M:%S.%f")
        print(f"{topic}:{date} = {value}")


def produce_frame(topic, producer, frame, callback=None, verbose=True):
    # Produce a message to the topic
    _, jpeg_frame = cv2.imencode(".jpg", frame)
    # Produce a message to the topic
    producer.produce(
        topic,
        key="cctv_07",
        value=jpeg_frame.tobytes(),
        callback=callback,
    )

    if verbose:
        _date = datetime.now()
        date = _date.strftime("%Y/%m/%d %H:%M:%S.%f")
        print(f"{topic}:{date} = {jpeg_frame}")


def delivery_report(err, msg):
    if err is not None:
        print("Message delivery failed: {}".format(err))
    else:
        print("Message delivered to {} [{}]".format(msg.topic(), msg.partition()))


def main():
    config = read_config()
    topic = "camera_07"

    # Video capture
    camera_id = "rtsp://210.99.70.120:1935/live/cctv007.stream"
    cap = cv2.VideoCapture(camera_id)

    # Callback function to handle delivery reports

    # creates a new producer instance
    producer = Producer(config)
    producer.flush()

    # produces a sample message
    i = 0
    while True:
        # Capture a frame
        ret, frame = cap.read()
        # Break the loop if the video capture is finished
        if not ret:
            break
        i += 1
        if i % 3 != 0:
            continue

        # Encode the frame to JPEG format
        _, jpeg_frame = cv2.imencode(".jpg", frame)

        produce(topic, producer, jpeg_frame, delivery_report)

        if i % (3 * 23) == 0:
            producer.flush()
    # send any outstanding or buffered messages to the Kafka broker

    # Release the video capture
    cap.release()

    # Close the producer
    producer.close()


if __name__ == "__main__":
    main()
