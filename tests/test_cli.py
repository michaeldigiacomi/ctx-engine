"""Tests for the ContextEngine CLI."""

import pytest
import subprocess
import sys
from unittest.mock import patch, MagicMock
from io import StringIO

from context_engine.cli import main
from context_engine import ContextEngine


class TestCLIArgs:
    """Test CLI argument parsing."""

    def test_cli_no_args_prints_help(self):
        """Test that running without args prints help."""
        with patch('sys.argv', ['ctx-engine']):
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                try:
                    main()
                except SystemExit:
                    pass

                output = mock_stdout.getvalue()
                assert "usage:" in output or "Commands" in output

    def test_save_command_parsing(self):
        """Test that save command parses arguments correctly."""
        with patch('sys.argv', [
            'ctx-engine', 'save', 'Test content',
            '--category', 'test',
            '--importance', '5.0',
            '--ttl', '7',
            '--tags', 'tag1', 'tag2'
        ]):
            with patch.object(ContextEngine, '__init__', return_value=None) as mock_init:
                with patch.object(ContextEngine, 'save', return_value='doc-id-123') as mock_save:
                    with patch.object(ContextEngine, 'close'):
                        try:
                            main()
                        except SystemExit:
                            pass

                        mock_save.assert_called_once()
                        call_kwargs = mock_save.call_args.kwargs
                        assert call_kwargs['content'] == 'Test content'
                        assert call_kwargs['category'] == 'test'
                        assert call_kwargs['importance'] == 5.0
                        assert call_kwargs['ttl_days'] == 7

    def test_search_command_parsing(self):
        """Test that search command parses arguments correctly."""
        with patch('sys.argv', [
            'ctx-engine', 'search', 'test query',
            '--limit', '5',
            '--min-similarity', '0.7',
            '--category', 'infrastructure'
        ]):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'search', return_value=[]) as mock_search:
                    with patch.object(ContextEngine, 'close'):
                        try:
                            main()
                        except SystemExit:
                            pass

                        mock_search.assert_called_once()
                        call_kwargs = mock_search.call_args.kwargs
                        assert call_kwargs['query'] == 'test query'
                        assert call_kwargs['limit'] == 5
                        assert call_kwargs['min_similarity'] == 0.7
                        assert call_kwargs['category'] == 'infrastructure'

    def test_list_command_parsing(self):
        """Test that list command parses arguments correctly."""
        with patch('sys.argv', [
            'ctx-engine', 'list',
            '--category', 'preference',
            '--limit', '20'
        ]):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'list', return_value=[]) as mock_list:
                    with patch.object(ContextEngine, 'close'):
                        try:
                            main()
                        except SystemExit:
                            pass

                        mock_list.assert_called_once()
                        call_kwargs = mock_list.call_args.kwargs
                        assert call_kwargs['category'] == 'preference'
                        assert call_kwargs['limit'] == 20

    def test_delete_command_parsing(self):
        """Test that delete command parses arguments correctly."""
        with patch('sys.argv', ['ctx-engine', 'delete', 'doc-id-123']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'delete', return_value=True) as mock_delete:
                    with patch.object(ContextEngine, 'close'):
                        try:
                            main()
                        except SystemExit:
                            pass

                        mock_delete.assert_called_once_with('doc-id-123')

    def test_cleanup_command(self):
        """Test that cleanup command works."""
        with patch('sys.argv', ['ctx-engine', 'cleanup']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'cleanup_expired', return_value=5) as mock_cleanup:
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                main()
                            except SystemExit:
                                pass

                            mock_cleanup.assert_called_once()
                            assert "5" in mock_stdout.getvalue()

    def test_init_command(self):
        """Test that init command works."""
        with patch('sys.argv', ['ctx-engine', 'init']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, '_ensure_initialized') as mock_init:
                    with patch.object(ContextEngine, 'close'):
                        try:
                            main()
                        except SystemExit:
                            pass

                        mock_init.assert_called_once()


class TestCLIOutput:
    """Test CLI output formatting."""

    def test_save_output_shows_doc_id(self):
        """Test that save command shows doc_id."""
        with patch('sys.argv', ['ctx-engine', 'save', 'Test memory']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'save', return_value='abc123'):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                main()
                            except SystemExit:
                                pass

                            assert "Saved:" in mock_stdout.getvalue()
                            assert "abc123" in mock_stdout.getvalue()

    def test_search_output_format(self):
        """Test search output format."""
        results = [
            {
                'similarity': 0.85,
                'category': 'infrastructure',
                'content': 'Deployed to Kubernetes cluster'
            }
        ]

        with patch('sys.argv', ['ctx-engine', 'search', 'k8s']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'search', return_value=results):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                main()
                            except SystemExit:
                                pass

                            output = mock_stdout.getvalue()
                            assert "[0.85]" in output
                            assert "[infrastructure]" in output

    def test_search_no_results(self):
        """Test search with no results message."""
        with patch('sys.argv', ['ctx-engine', 'search', 'query']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'search', return_value=[]):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                main()
                            except SystemExit:
                                pass

                            assert "No results found" in mock_stdout.getvalue()

    def test_list_output_format(self):
        """Test list output format."""
        from datetime import datetime

        memories = [
            {
                'created_at': datetime(2024, 1, 15, 10, 30, 0),
                'category': 'preference',
                'content': 'User prefers dark mode'
            }
        ]

        with patch('sys.argv', ['ctx-engine', 'list']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'list', return_value=memories):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                main()
                            except SystemExit:
                                pass

                            output = mock_stdout.getvalue()
                            assert "2024-01-15" in output
                            assert "[preference]" in output

    def test_delete_success_output(self):
        """Test delete success message."""
        with patch('sys.argv', ['ctx-engine', 'delete', 'doc-id']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'delete', return_value=True):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                main()
                            except SystemExit:
                                pass

                            assert "Deleted" in mock_stdout.getvalue()

    def test_delete_not_found_output(self):
        """Test delete not found message."""
        with patch('sys.argv', ['ctx-engine', 'delete', 'nonexistent']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'delete', return_value=False):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                main()
                            except SystemExit:
                                pass

                            assert "Not found" in mock_stdout.getvalue()

    def test_get_context_output(self):
        """Test get-context output."""
        context = "[infrastructure] Deployed to k8s"

        with patch('sys.argv', ['ctx-engine', 'get-context', 'deployment']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'get_context', return_value=context):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                main()
                            except SystemExit:
                                pass

                            assert context in mock_stdout.getvalue()

    def test_get_context_empty_output(self):
        """Test get-context with no results."""
        with patch('sys.argv', ['ctx-engine', 'get-context', 'query']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'get_context', return_value=''):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            try:
                                main()
                            except SystemExit:
                                pass

                            assert "(no context found)" in mock_stdout.getvalue()


class TestCLIErrorHandling:
    """Test CLI error handling."""

    def test_init_failure_prints_error(self):
        """Test that initialization failure shows error."""
        with patch('sys.argv', ['ctx-engine', 'save', 'test']):
            with patch.object(ContextEngine, '__init__', side_effect=Exception("DB error")):
                with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
                    with patch('sys.stdout', new_callable=StringIO):
                        with pytest.raises(SystemExit) as exc_info:
                            main()

                        assert exc_info.value.code == 1

    def test_command_error_prints_message(self):
        """Test that command errors are printed."""
        with patch('sys.argv', ['ctx-engine', 'save', 'test']):
            with patch.object(ContextEngine, '__init__', return_value=None):
                with patch.object(ContextEngine, 'save', side_effect=Exception("Save failed")):
                    with patch.object(ContextEngine, 'close'):
                        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                            with pytest.raises(SystemExit) as exc_info:
                                main()

                            assert "Error:" in mock_stdout.getvalue()
                            assert exc_info.value.code == 1
