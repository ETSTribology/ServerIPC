import abc
from typing import Any, Dict, List

class BoardBase(abc.ABC):
    """
    Abstract base class for monitoring backends.
    """

    @abc.abstractmethod
    def setup_dashboard(self, dashboard_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new dashboard in the monitoring system.

        Args:
            dashboard_config: Configuration dictionary for the dashboard.

        Returns:
            A dictionary containing details of the created dashboard.
        """
        pass

    @abc.abstractmethod
    def add_panel(self, dashboard_id: str, panel_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add a new panel to an existing dashboard.

        Args:
            dashboard_id: The ID of the dashboard to add the panel to.
            panel_config: Configuration dictionary for the panel.

        Returns:
            A dictionary containing details of the created panel.
        """
        pass

    @abc.abstractmethod
    def update_panel(self, dashboard_id: str, panel_id: str, panel_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an existing panel in a dashboard.

        Args:
            dashboard_id: The ID of the dashboard containing the panel.
            panel_id: The ID of the panel to update.
            panel_config: Updated configuration dictionary for the panel.

        Returns:
            A dictionary containing details of the updated panel.
        """
        pass

    @abc.abstractmethod
    def delete_dashboard(self, dashboard_id: str) -> None:
        """
        Delete a dashboard from the monitoring system.

        Args:
            dashboard_id: The ID of the dashboard to delete.
        """
        pass

    @abc.abstractmethod
    def list_dashboards(self) -> List[Dict[str, Any]]:
        """
        Retrieve a list of all dashboards.

        Returns:
            A list of dictionaries, each representing a dashboard.
        """
        pass
