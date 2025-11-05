#!/usr/bin/env python

import json
import sys
import time

import pika
from pika.adapters.blocking_connection import BlockingChannel
from post import MisinformationReport, MisinfoState, post_from_json


def main():
    """
    1. initialise the model, such that it will run in the background
    2. make a function which will use the AI model to process the post
    3. once post has finished processing, acknowledge to rabbitmq that we successfully handled the response.
    4. send ai report to rabbitmq, so api can store it in the database
    5. connect to rabbitmq, initialise the queue to listen on, and the callback function
    """
    print("initialising AI model")
    # 1 [init your model here]

    # 2 [your model will be used here]
    def on_new_post(channel: BlockingChannel, method, properties, body: str):
        # body will contain json from api
        post = post_from_json(body)
        print(f"staring ai task. post id: {post.id}")

        time.sleep(5)
        ai_model_result = MisinfoState.TRUE  # OR MisinfoState.FAKE

        report = MisinformationReport(post.id, ai_model_result)
        print(f"completed ai task. waiting for next task")

        # 3 [tell rabbitmq we finished]
        channel.basic_ack(delivery_tag=method.delivery_tag)

        # 4 [tell rabbitmq result of ai]
        report_json = json.dumps(report.to_dict())
        print(report_json)
        channel.basic_publish(
            exchange="",
            routing_key="misinfo/output",
            body=report_json,
        )

    # 5 [connect to rabbitmq]
    print("connecting to rabbitmq")
    with pika.BlockingConnection(pika.ConnectionParameters("localhost")) as connection:
        with connection.channel() as channel:
            channel.queue_declare(queue="misinfo/input")
            channel.queue_declare(queue="misinfo/output")

            channel.basic_consume(
                queue="misinfo/input", on_message_callback=on_new_post
            )

            print("consuming queue 'misinfo/input'")
            channel.start_consuming()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("interrupt")
        sys.exit(0)


# EXAMPLE
#
# You should copy/paste this code to connect it to the API.
#
# The API shall send all new posts to a RabbitMQ (message queue)
# Once the AI has gotten it's result, it shall send it to the MQ,
# which the API will listen on, and sync with the database
