import yt_dlp
import os
import re
from collections import Counter


class ErrorTrackingLogger:
    """Custom logger that intercepts yt_dlp errors and tallies them by type."""

    ERROR_PATTERNS = [
        (r'Video unavailable', 'Video unavailable'),
        (r'Private video', 'Private video'),
        (r'This video has been removed', 'Video removed'),
        (r'Sign in to confirm', 'Sign-in required'),
        (r'HTTP Error 403', 'HTTP 403 Forbidden'),
        (r'HTTP Error 429', 'HTTP 429 Too Many Requests'),
        (r'not available in your country', 'Geo-restricted'),
        (r'copyright', 'Copyright claim'),
        (r'is not a valid URL', 'Invalid URL'),
        (r'Incomplete.*data', 'Incomplete data'),
        (r'Unable to extract', 'Extraction failed'),
        (r'urlopen error', 'Network error'),
        (r'timed?\s*out', 'Timeout'),
        (r'members-only', 'Members-only content'),
        (r'age', 'Age-restricted'),
    ]

    def __init__(self):
        self.error_counts = Counter()

    def debug(self, msg):
        print(msg)

    def info(self, msg):
        print(msg)

    def warning(self, msg):
        print(f'WARNING: {msg}')

    def error(self, msg):
        print(f'ERROR: {msg}')
        error_type = self._classify_error(msg)
        self.error_counts[error_type] += 1

    def _classify_error(self, msg):
        for pattern, label in self.ERROR_PATTERNS:
            if re.search(pattern, msg, re.IGNORECASE):
                return label
        # Use the first line (truncated) for unrecognized errors
        first_line = msg.split('\n')[0].strip()
        return first_line[:80] if len(first_line) > 80 else first_line

    def print_summary(self):
        if not self.error_counts:
            print('\nAll downloads completed successfully!')
            return
        total = sum(self.error_counts.values())
        print(f'\n{"=" * 50}')
        print(f'Download Failure Summary ({total} total failures)')
        print(f'{"=" * 50}')
        for error_type, count in self.error_counts.most_common():
            print(f'  {error_type}: {count}')
        print(f'{"=" * 50}')


def download_playlists(file_path, download_dir):
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    cookie_path = os.path.join(os.path.dirname(file_path), 'cookies.txt')
    logger = ErrorTrackingLogger()

    ydl_opts = {
        'paths': {'home': download_dir},
        'format': 'bestvideo[ext=mp4]/bestvideo/best',
        'merge_output_format': 'mp4',
        'outtmpl': '%(playlist_title)s/%(playlist_index)s - %(title)s.%(ext)s',
        # Use cookies to avoid bot detection / rate limiting
        'cookiefile': cookie_path,
        # Enable node JS runtime to solve YouTube's n-parameter challenge
        'js_runtimes': {'node': {}},
        # web client supports cookies for authenticated downloads
        'extractor_args': {
            'youtube': {
                'player_client': ['web'],
            }
        },
        # Sleep between downloads to avoid rate limiting
        'sleep_interval': 30,
        'max_sleep_interval': 120,
        # Retry on transient network errors instead of crashing
        'retries': 10,
        'fragment_retries': 10,
        'socket_timeout': 30,
        # Skip failed videos and continue with the rest
        'ignoreerrors': True,
        # Resume partially downloaded files
        'continuedl': True,
        # Track completed downloads so re-running skips them
        'download_archive': os.path.join(download_dir, '.downloaded.txt'),
        # Custom logger to track errors
        'logger': logger,
    }

    with open(file_path, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]
    print(urls)

    print('start downloading...')
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download(urls)

    logger.print_summary()


if __name__ == "__main__":
    my_path = '/raid/robot/youtube_videos'
    download_playlists('citywalk_playlists.txt', my_path)
