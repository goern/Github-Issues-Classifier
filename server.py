# __init__.py
import logging
import sys
from tornado.httpserver import HTTPServer
from tornado.options import define, options
from tornado.web import Application
from tornado.web import RequestHandler
from tornado.ioloop import IOLoop
from tornado.escape import json_decode
from __init__ import __version__

from label_bot import models

import os

os.environ["WANDB_SILENT"] = "true"

define("port", default=8888, help="port to listen on")

_LOGGER = logging.getLogger(__name__)


def init_models():
    """
    Initializes the trained model.
    """
    global BOT
    BOT = models.Bot(use_head=True)


class PredictHandler(RequestHandler):
    """
    Take the title and body of an Issue and give the classification dict.
    """

    async def predict(self, title, body):
        """
        Returns the prediction scores.

        args:
            title : the titles of the issues.
            body  : the bodies of the issues.
        """
        return BOT.predict(title, body)[0]

    async def post(self):
        req = json_decode(self.request.body)
        title = req.get("title", "")
        body = req.get("body", "")

        b_score, q_score, e_score = await self.predict(title, body)
        response = {
            "title": title,
            "body": body,
            "bug": str(b_score),
            "question": str(q_score),
            "enhancement": str(e_score),
        }
        self.write(response)
        self.finish()


class MainHandler(RequestHandler):
    """
    Give the application name and the current version.
    """

    async def get(self):
        response = {"name": "Thoth Label Classifier", "version": __version__}
        self.write(response)
        self.finish()


def main():
    """Construct and serve the tornado application."""
    app = Application(
        [
            (r"/", MainHandler),
            (r"/predict", PredictHandler),
        ]
    )
    http_server = HTTPServer(app)
    http_server.listen(options.port)
    # Finally start the server
    _LOGGER.info("Listening on http://localhost:%i" % options.port)
    IOLoop.current().start()


if __name__ == "__main__":
    init_models()
    main()
    sys.exit(1)
