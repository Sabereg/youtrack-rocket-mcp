"""
Unit tests for issue attachments: listing, downloading and the MCP tools on top of them.
"""

import json
from unittest.mock import AsyncMock

import httpx
import pytest

from youtrack_rocket_mcp.api.client import ResourceNotFoundError, YouTrackClient
from youtrack_rocket_mcp.api.resources.issues import IssuesClient
from youtrack_rocket_mcp.tools.issues import IssueTools

ATTACHMENTS = [
    {
        'id': '78-1',
        'name': 'mockup.png',
        'url': '/api/files/78-1?sign=abc',
        'mimeType': 'image/png',
        'size': 3,
        '$type': 'IssueAttachment',
    },
    {
        'id': '78-2',
        'name': 'spec.pdf',
        'url': '/api/files/78-2?sign=def',
        'mimeType': 'application/pdf',
        'size': 5,
        '$type': 'IssueAttachment',
    },
]


def make_client(base_url: str, handler) -> YouTrackClient:
    client = YouTrackClient(base_url=base_url, api_token='test-token')
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


@pytest.fixture
def issue_tools(mock_client):
    """IssueTools wired to a mock API client, bypassing config-based construction."""
    tools = IssueTools.__new__(IssueTools)
    tools.client = mock_client
    tools.issues_api = IssuesClient(mock_client)
    return tools


@pytest.mark.asyncio
async def test_get_issue_attachments_requests_fields(mock_client):
    """Attachments are requested from the issue endpoint with name, url and comment link."""
    mock_client.get.return_value = ATTACHMENTS

    result = await IssuesClient(mock_client).get_issue_attachments('TEST-1')

    endpoint = mock_client.get.call_args[0][0]
    assert endpoint.startswith('issues/TEST-1/attachments?fields=')
    for field in ('name', 'url', 'mimeType', 'size', 'comment(id)'):
        assert field in endpoint
    assert result == ATTACHMENTS


@pytest.mark.parametrize(
    ('base_url', 'file_url', 'expected'),
    [
        ('https://yt.example.com/api', '/api/files/1?sign=x', 'https://yt.example.com/api/files/1?sign=x'),
        (
            'https://example.com/youtrack/api',
            '/youtrack/api/files/1?sign=x',
            'https://example.com/youtrack/api/files/1?sign=x',
        ),
        ('https://yt.example.com/api', 'https://cdn.example.com/f/1', 'https://cdn.example.com/f/1'),
    ],
)
@pytest.mark.asyncio
async def test_download_resolves_url_against_server(base_url, file_url, expected):
    """Server-relative attachment URLs are resolved against the instance, including a context path."""
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return httpx.Response(200, content=b'data')

    client = make_client(base_url, handler)

    content = await client.download(file_url)

    assert content == b'data'
    assert requested == [expected]


@pytest.mark.asyncio
async def test_download_maps_not_found():
    """A missing file raises ResourceNotFoundError."""
    client = make_client('https://yt.example.com/api', lambda _: httpx.Response(404))

    with pytest.raises(ResourceNotFoundError):
        await client.download('/api/files/1')


@pytest.mark.asyncio
async def test_download_attachment_by_name_writes_file(issue_tools, mock_client, tmp_path):
    """The attachment chosen by name is saved into dest_dir and its path is returned."""
    mock_client.get.return_value = ATTACHMENTS
    mock_client.download = AsyncMock(return_value=b'png')

    result = json.loads(await issue_tools.download_attachment('TEST-1', 'mockup.png', str(tmp_path)))

    mock_client.download.assert_awaited_once_with('/api/files/78-1?sign=abc')
    assert result['path'] == str(tmp_path / 'mockup.png')
    assert (tmp_path / 'mockup.png').read_bytes() == b'png'


@pytest.mark.asyncio
async def test_download_attachment_by_id(issue_tools, mock_client, tmp_path):
    """The attachment can be chosen by its ID."""
    mock_client.get.return_value = ATTACHMENTS
    mock_client.download = AsyncMock(return_value=b'%PDF-')

    result = json.loads(await issue_tools.download_attachment('TEST-1', '78-2', str(tmp_path)))

    assert result['name'] == 'spec.pdf'
    assert (tmp_path / 'spec.pdf').read_bytes() == b'%PDF-'


@pytest.mark.asyncio
async def test_download_attachment_unknown_lists_available(issue_tools, mock_client, tmp_path):
    """An unknown attachment yields an error listing what is available."""
    mock_client.get.return_value = ATTACHMENTS
    mock_client.download = AsyncMock()

    result = json.loads(await issue_tools.download_attachment('TEST-1', 'missing.txt', str(tmp_path)))

    assert 'error' in result
    assert [a['name'] for a in result['available']] == ['mockup.png', 'spec.pdf']
    mock_client.download.assert_not_awaited()


@pytest.mark.asyncio
async def test_download_attachment_ambiguous_name_requires_id(issue_tools, mock_client, tmp_path):
    """Two attachments with the same name are not guessed between."""
    mock_client.get.return_value = [ATTACHMENTS[0], {**ATTACHMENTS[0], 'id': '78-3'}]
    mock_client.download = AsyncMock()

    result = json.loads(await issue_tools.download_attachment('TEST-1', 'mockup.png', str(tmp_path)))

    assert [m['id'] for m in result['matches']] == ['78-1', '78-3']
    mock_client.download.assert_not_awaited()


@pytest.mark.asyncio
async def test_download_attachment_name_cannot_escape_dest_dir(issue_tools, mock_client, tmp_path):
    """A path-like attachment name is reduced to its base name inside dest_dir."""
    mock_client.get.return_value = [{**ATTACHMENTS[0], 'name': '../../evil.png'}]
    mock_client.download = AsyncMock(return_value=b'x')
    dest = tmp_path / 'dest'

    result = json.loads(await issue_tools.download_attachment('TEST-1', '../../evil.png', str(dest)))

    assert result['path'] == str(dest / 'evil.png')
    assert not (tmp_path.parent / 'evil.png').exists()
