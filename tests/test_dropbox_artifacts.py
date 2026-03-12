from dropbox.exceptions import ApiError
from dropbox.files import GetMetadataError, LookupError

from src.ops.dropbox_artifacts import DropboxArtifactsClient


def test_is_not_found_only_matches_lookup_not_found() -> None:
    not_found = ApiError(
        request_id="req-1",
        error=GetMetadataError.path(LookupError.not_found),
        user_message_text=None,
        user_message_locale=None,
    )
    not_folder = ApiError(
        request_id="req-2",
        error=GetMetadataError.path(LookupError.not_folder),
        user_message_text=None,
        user_message_locale=None,
    )

    assert DropboxArtifactsClient._is_not_found(not_found) is True
    assert DropboxArtifactsClient._is_not_found(not_folder) is False
