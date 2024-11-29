try:
    from .folder import currently_dir, artifacts_dir, media_dir
except:
    from folder import currently_dir, artifacts_dir, media_dir


from kot import KOT



kot_db_ = KOT("gca", folder=artifacts_dir, enable_hashing=True)





