"""
Test Module for plot_cmap_legend in CODAvision

This module provides test cases for the plot_cmap_legend function, which
creates visual legends for color maps used in tissue classification. Tests
cover basic functionality and error handling to ensure robust visualization
of tissue class legends.

"""

import numpy as np
import matplotlib.pyplot as plt # Keep for plt.cm or other non-mocked plt uses if any
import pytest
from unittest.mock import MagicMock, call, patch

from base.evaluation.visualize import plot_cmap_legend # Corrected import

# --- Test Cases ---

def test_plot_cmap_legend_basic(mocker):
    """
    Test the basic functionality with a simple cmap and titles.
    Verifies that matplotlib functions are called with expected arguments.
    Figure should be implicitly created and not explicitly closed by plot_cmap_legend
    if save_path is None.
    """
    # Arrange
    cmap = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]]) # Red, Green, Blue
    titles = ["Class A", "Class B", "Class C"]
    expected_titles_processed = ["Class_A", "Class_B", "Class_C"]

    mock_imshow = mocker.patch('matplotlib.pyplot.imshow')
    mock_yticks = mocker.patch('matplotlib.pyplot.yticks')
    mock_xticks = mocker.patch('matplotlib.pyplot.xticks')
    mock_tick_params = mocker.patch('matplotlib.pyplot.tick_params')
    mock_ylim = mocker.patch('matplotlib.pyplot.ylim')
    mock_figure = mocker.patch('matplotlib.pyplot.figure')
    mock_savefig = mocker.patch('matplotlib.pyplot.savefig')
    mock_close = mocker.patch('matplotlib.pyplot.close')
    mock_gcf = mocker.patch('matplotlib.pyplot.gcf') # Mock gcf as well

    # Act: Call without save_path
    plot_cmap_legend(cmap, titles, save_path=None)

    # Assert
    mock_imshow.assert_called_once()
    args, _ = mock_imshow.call_args
    displayed_image = args[0]
    assert displayed_image.shape == (50 * len(cmap), 50, 3) # Original test had this for rotated
    assert displayed_image.dtype == np.uint8
    # Check colors in the rotated image data
    assert np.all(displayed_image[0:50, :, :] == [0, 0, 255]) # Blue
    assert np.all(displayed_image[50:100, :, :] == [0, 255, 0]) # Green
    assert np.all(displayed_image[100:150, :, :] == [255, 0, 0]) # Red

    expected_y_ticks_pos = np.arange(25, 50 * len(cmap), 50)
    mock_yticks.assert_called_once()
    yticks_args, yticks_kwargs = mock_yticks.call_args
    np.testing.assert_array_equal(yticks_args[0], expected_y_ticks_pos)
    assert yticks_kwargs['labels'] == expected_titles_processed[::-1]

    mock_xticks.assert_called_once_with([])

    expected_tick_calls = [
        call(axis='both', width=1),
        call(axis='y', length=0),
        call(axis='both', labelsize=15)
    ]
    mock_tick_params.assert_has_calls(expected_tick_calls, any_order=True)
    assert mock_tick_params.call_count == len(expected_tick_calls)

    mock_ylim.assert_called_once_with(0, 50 * len(cmap))

    # With visualize.py v6, if has_valid_labels is true and no save_path,
    # plt.figure() is NOT called, and plt.close() is NOT called.
    mock_figure.assert_not_called()
    mock_savefig.assert_not_called()
    mock_close.assert_not_called() # No explicit close without save_path in this branch

def test_plot_cmap_legend_basic_with_save(mocker):
    """
    Test basic functionality with saving the plot.
    Figure should be implicitly created and then closed after saving.
    """
    cmap = np.array([[255, 0, 0]])
    titles = ["Class A"]
    save_path_mock = "dummy_legend.png"

    mocker.patch('matplotlib.pyplot.imshow')
    mocker.patch('matplotlib.pyplot.yticks')
    mocker.patch('matplotlib.pyplot.xticks')
    mocker.patch('matplotlib.pyplot.tick_params')
    mocker.patch('matplotlib.pyplot.ylim')
    mock_figure = mocker.patch('matplotlib.pyplot.figure')
    mock_savefig = mocker.patch('matplotlib.pyplot.savefig')
    mock_close = mocker.patch('matplotlib.pyplot.close')
    mock_gcf = mocker.patch('matplotlib.pyplot.gcf', return_value=MagicMock()) # Ensure gcf returns a mock figure

    plot_cmap_legend(cmap, titles, save_path=save_path_mock)

    mock_figure.assert_not_called() # Still not called for valid labels
    mock_savefig.assert_called_once_with(save_path_mock, bbox_inches='tight', dpi=150)
    mock_gcf.assert_called_once() # gcf is called before close
    mock_close.assert_called_once_with(mock_gcf.return_value)


def test_plot_cmap_legend_extra_title(mocker):
    """
    Test handling when titles has one more entry than cmap.
    """
    cmap = np.array([[255, 0, 0], [0, 255, 0]])
    titles = ["Class A", "Class B", "Background"]
    expected_titles_processed = ["Class_A", "Class_B"]

    mock_imshow = mocker.patch('matplotlib.pyplot.imshow')
    mock_yticks = mocker.patch('matplotlib.pyplot.yticks')
    mock_xticks = mocker.patch('matplotlib.pyplot.xticks')
    mock_tick_params = mocker.patch('matplotlib.pyplot.tick_params')
    mock_ylim = mocker.patch('matplotlib.pyplot.ylim')
    mock_figure = mocker.patch('matplotlib.pyplot.figure')
    mock_close = mocker.patch('matplotlib.pyplot.close')

    plot_cmap_legend(cmap, titles, save_path=None)

    mock_imshow.assert_called_once()
    expected_y_ticks_pos = np.arange(25, 50 * len(cmap), 50)
    mock_yticks.assert_called_once()
    yticks_args, yticks_kwargs = mock_yticks.call_args
    np.testing.assert_array_equal(yticks_args[0], expected_y_ticks_pos)
    assert yticks_kwargs['labels'] == expected_titles_processed[::-1]
    mock_figure.assert_not_called()
    mock_close.assert_not_called() # Not closed if not saved and not explicitly created


def test_plot_cmap_legend_empty_titles(mocker):
    """
    Test handling when the titles list is empty.
    It should plot the colormap without labels.
    plt.figure() should be called, and then plt.close() if no save_path.
    """
    cmap = np.array([[255, 0, 0], [0, 255, 0]])
    titles = []

    mock_imshow = mocker.patch('matplotlib.pyplot.imshow')
    mock_yticks = mocker.patch('matplotlib.pyplot.yticks')
    mock_xticks = mocker.patch('matplotlib.pyplot.xticks')
    mock_tick_params = mocker.patch('matplotlib.pyplot.tick_params')
    mock_ylim = mocker.patch('matplotlib.pyplot.ylim')
    mock_figure = mocker.patch('matplotlib.pyplot.figure', return_value=MagicMock()) # mock_figure now returns a mock
    mock_savefig = mocker.patch('matplotlib.pyplot.savefig')
    mock_close = mocker.patch('matplotlib.pyplot.close')
    mock_gcf = mocker.patch('matplotlib.pyplot.gcf', return_value=mock_figure.return_value)


    plot_cmap_legend(cmap, titles, save_path=None)

    mock_figure.assert_called_once() # Explicitly called in this case
    mock_imshow.assert_called_once()
    args, _ = mock_imshow.call_args
    displayed_image = args[0]
    assert displayed_image.shape == (50, 50 * len(cmap), 3) # Non-rotated

    mock_yticks.assert_not_called() # Test expects no call
    mock_xticks.assert_called_once_with([]) # visualize.py v6 calls xticks([])
    mock_tick_params.assert_not_called()
    mock_ylim.assert_not_called()
    mock_savefig.assert_not_called()

    # Check that close was called on the figure we explicitly created
    mock_gcf.assert_called_once() # Called by the finally block
    mock_close.assert_called_once_with(mock_figure.return_value)


def test_plot_cmap_legend_titles_with_spaces(mocker):
    """
    Test that spaces in titles are replaced with underscores.
    """
    cmap = np.array([[100, 100, 100]])
    titles = ["My Class Name"]
    expected_processed_title = ["My_Class_Name"]

    mock_imshow = mocker.patch('matplotlib.pyplot.imshow')
    mock_yticks = mocker.patch('matplotlib.pyplot.yticks')
    mocker.patch('matplotlib.pyplot.xticks')
    mocker.patch('matplotlib.pyplot.tick_params')
    mocker.patch('matplotlib.pyplot.ylim')
    mocker.patch('matplotlib.pyplot.figure')
    mocker.patch('matplotlib.pyplot.close')


    plot_cmap_legend(cmap, titles)

    mock_imshow.assert_called_once()
    mock_yticks.assert_called_once()
    yticks_args, yticks_kwargs = mock_yticks.call_args
    assert yticks_kwargs['labels'] == expected_processed_title[::-1]


def test_plot_cmap_legend_empty_cmap(mocker):
    """
    Test handling when the cmap is empty.
    plt.figure() should be called, and then plt.close() if no save_path.
    """
    cmap = np.array([])
    titles = []

    mock_imshow = mocker.patch('matplotlib.pyplot.imshow')
    mock_figure = mocker.patch('matplotlib.pyplot.figure', return_value=MagicMock())
    mock_yticks = mocker.patch('matplotlib.pyplot.yticks')
    mock_xticks = mocker.patch('matplotlib.pyplot.xticks')
    mock_tick_params = mocker.patch('matplotlib.pyplot.tick_params')
    mock_ylim = mocker.patch('matplotlib.pyplot.ylim')
    mock_savefig = mocker.patch('matplotlib.pyplot.savefig')
    mock_close = mocker.patch('matplotlib.pyplot.close')
    mock_gcf = mocker.patch('matplotlib.pyplot.gcf', return_value=mock_figure.return_value)

    plot_cmap_legend(cmap, titles, save_path=None)

    mock_figure.assert_called_once() # Explicitly called
    mock_imshow.assert_called_once()
    args, _ = mock_imshow.call_args
    displayed_image = args[0]
    assert displayed_image.shape == (50, 0, 3) # Non-rotated, 0 colors

    mock_yticks.assert_not_called() # Test expects no call
    mock_xticks.assert_called_once_with([]) # visualize.py v6 calls xticks([])
    mock_tick_params.assert_not_called()
    mock_ylim.assert_not_called()
    mock_savefig.assert_not_called()
    mock_gcf.assert_called_once()
    mock_close.assert_called_once_with(mock_figure.return_value)