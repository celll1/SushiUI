"""Non-ASCII uploaded filenames must survive into `params` intact.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/upload_filename_recovery_test.py -v

Starlette's multipart parser decodes a part's `filename` with the request's
declared charset (utf-8 unless the request itself overrides it), falling back
to latin-1 on a decode failure (`starlette.formparsers._user_safe_decode`).
That fallback never loses bytes -- latin-1 maps every byte 1:1 -- so a
filename mis-decoded that way is losslessly reversible by re-encoding as
latin-1 and decoding as UTF-8. `recover_upload_filename` does exactly that,
and is a no-op on names that were already decoded correctly.

MUTANT (see the last test): "skip recovery" -- store `UploadFile.filename`
straight into `params`, as the code did before this fix. It passes every
default-charset request (this repo's real traffic decodes correctly already,
which is why the round-trip test below forces the latin-1 fallback via an
explicit `charset=` parameter -- the one condition that actually exercises
the failure mode this fix targets) but fails the moment a request lands in
that fallback, storing garbled text with no warning anywhere.
"""

import asyncio
import os
import sys
from typing import List, Optional

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BACKEND = os.path.join(_REPO, "backend")
for _p in (_REPO, _BACKEND):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import unittest  # noqa: E402

import httpx  # noqa: E402
from fastapi import FastAPI, File, UploadFile  # noqa: E402

from utils.upload_names import recover_upload_filename  # noqa: E402

NAME = "生成画像1.png"


class RecoverUploadFilenameUnitTest(unittest.TestCase):
    def test_ascii_name_is_unchanged(self):
        self.assertEqual(recover_upload_filename("photo.png"), "photo.png")

    def test_none_and_empty_are_unchanged(self):
        self.assertIsNone(recover_upload_filename(None))
        self.assertEqual(recover_upload_filename(""), "")

    def test_mis_decoded_latin1_is_recovered(self):
        # The exact failure mode: correctly-decoded UTF-8 bytes reinterpreted
        # one byte per latin-1 code point.
        mojibake = NAME.encode("utf-8").decode("latin-1")
        self.assertEqual(recover_upload_filename(mojibake), NAME)

    def test_already_correct_unicode_is_left_alone(self):
        # NAME itself contains code points > U+00FF, so encoding it as
        # latin-1 raises -- the guard that stops this from double-mangling
        # a name that was already decoded correctly.
        self.assertEqual(recover_upload_filename(NAME), NAME)

    def test_genuine_replacement_characters_are_not_claimed_recovered(self):
        # A name that already lost bytes (e.g. a strict/`errors="replace"`
        # decode upstream) is unrecoverable -- U+FFFD does not roundtrip
        # through latin-1 encode. left untouched, not silently "fixed".
        lossy = "��1.png"
        self.assertEqual(recover_upload_filename(lossy), lossy)


class RealUploadPathRoundTripTest(unittest.TestCase):
    """POSTs an actual multipart body through FastAPI's own File() parameter,
    the same shape /generate/ref2vid's `reference_images` uses, then applies
    `recover_upload_filename` exactly as routes.py does.
    """

    def _post_multipart(self, content_type_charset: Optional[str]):
        app = FastAPI()

        @app.post("/upload")
        async def upload(reference_images: List[UploadFile] = File([])):
            return {
                "reference_images": [
                    recover_upload_filename(f.filename) for f in reference_images
                ]
            }

        name_bytes = NAME.encode("utf-8")
        boundary = "----TestBoundaryXYZ"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="reference_images"; filename="'
        ).encode() + name_bytes + (
            '"\r\nContent-Type: image/png\r\n\r\nFAKEPNGDATA'
            f"\r\n--{boundary}--\r\n"
        ).encode()

        content_type = f"multipart/form-data; boundary={boundary}"
        if content_type_charset:
            content_type += f"; charset={content_type_charset}"

        async def run():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.post(
                    "/upload", content=body, headers={"Content-Type": content_type}
                )

        return asyncio.run(run())

    def test_default_charset_round_trips_correctly(self):
        # This repo's real traffic: no explicit charset, Starlette defaults to
        # utf-8, recover_upload_filename is a no-op on the already-correct name.
        resp = self._post_multipart(content_type_charset=None)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["reference_images"], [NAME])

    def test_latin1_declared_request_still_recovers_the_name(self):
        # Forces Starlette's OWN decode of `filename` to use latin-1 (a
        # client, proxy, or future default that declares this charset would
        # hit the exact fallback `_user_safe_decode` takes on a decode
        # failure). recover_upload_filename must undo it.
        resp = self._post_multipart(content_type_charset="latin-1")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["reference_images"], [NAME])

    def test_mutant_skip_recovery_fails_the_latin1_case(self):
        """MUTANT: store UploadFile.filename directly, without
        recover_upload_filename -- the code before this fix. Proves the
        fixed test actually distinguishes correct from broken behaviour.
        """
        app = FastAPI()

        @app.post("/upload")
        async def upload(reference_images: List[UploadFile] = File([])):
            return {"reference_images": [f.filename for f in reference_images]}

        name_bytes = NAME.encode("utf-8")
        boundary = "----TestBoundaryXYZ"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="reference_images"; filename="'
        ).encode() + name_bytes + (
            '"\r\nContent-Type: image/png\r\n\r\nFAKEPNGDATA'
            f"\r\n--{boundary}--\r\n"
        ).encode()
        content_type = f"multipart/form-data; boundary={boundary}; charset=latin-1"

        async def run():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.post(
                    "/upload", content=body, headers={"Content-Type": content_type}
                )

        resp = asyncio.run(run())
        self.assertNotEqual(resp.json()["reference_images"], [NAME])


if __name__ == "__main__":
    unittest.main()
