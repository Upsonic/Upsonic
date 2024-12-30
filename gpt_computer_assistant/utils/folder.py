import os

currently_dir = os.path.dirname(os.path.abspath(__file__))
artifacts_dir = os.path.join(currently_dir, "artifacts")
media_dir = os.path.join(currently_dir, "media")

if not os.path.exists(artifacts_dir):
    os.makedirs(artifacts_dir)