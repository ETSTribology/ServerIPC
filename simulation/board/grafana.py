import logging
from typing import Any, Dict, List

import requests
from board.base import BoardBase


class Grafana(BoardBase):
    """
    Grafana implementation of MonitoringBase using Grafana HTTP API.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Grafana monitoring client.

        Args:
            config: Configuration dictionary containing Grafana settings.
                Required keys:
                    - url: Base URL of Grafana (e.g., "http://localhost:3000")
                    - api_key: Grafana API key with appropriate permissions
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.url = config.get("url", "http://localhost:3000")
        self.api_key = config.get("api_key")
        if not self.api_key:
            raise ValueError("Grafana API key must be provided in the config.")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self.logger.info("Initialized GrafanaMonitoring client.")

    def setup_dashboard(self, dashboard_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new dashboard in Grafana.

        Args:
            dashboard_config: Configuration dictionary for the dashboard.

        Returns:
            A dictionary containing details of the created dashboard.
        """
        endpoint = f"{self.url}/api/dashboards/db"
        payload = {"dashboard": dashboard_config, "folderId": 0, "overwrite": False}
        self.logger.debug(f"Creating dashboard with payload: {payload}")
        response = requests.post(endpoint, headers=self.headers, json=payload)
        if response.status_code == 200:
            self.logger.info("Dashboard created successfully.")
            return response.json()
        else:
            self.logger.error(f"Failed to create dashboard: {response.text}")
            response.raise_for_status()

    def add_panel(self, dashboard_id: str, panel_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add a new panel to an existing dashboard.

        Args:
            dashboard_id: The UID of the dashboard to add the panel to.
            panel_config: Configuration dictionary for the panel.

        Returns:
            A dictionary containing details of the created panel.
        """
        # Retrieve existing dashboard
        dashboard = self.get_dashboard_by_uid(dashboard_id)
        if not dashboard:
            raise ValueError(f"Dashboard with UID '{dashboard_id}' not found.")

        dashboard_body = dashboard["dashboard"]
        panels = dashboard_body.get("panels", [])
        panels.append(panel_config)
        dashboard_body["panels"] = panels

        # Update dashboard
        payload = {
            "dashboard": dashboard_body,
            "folderId": dashboard["meta"]["folderId"],
            "overwrite": True,
        }
        self.logger.debug(f"Updating dashboard '{dashboard_id}' with new panel.")
        endpoint = f"{self.url}/api/dashboards/db"
        response = requests.post(endpoint, headers=self.headers, json=payload)
        if response.status_code == 200:
            self.logger.info("Panel added successfully.")
            return response.json()
        else:
            self.logger.error(f"Failed to add panel: {response.text}")
            response.raise_for_status()

    def update_panel(
        self, dashboard_id: str, panel_id: int, panel_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update an existing panel in a dashboard.

        Args:
            dashboard_id: The UID of the dashboard containing the panel.
            panel_id: The ID of the panel to update.
            panel_config: Updated configuration dictionary for the panel.

        Returns:
            A dictionary containing details of the updated dashboard.
        """
        # Retrieve existing dashboard
        dashboard = self.get_dashboard_by_uid(dashboard_id)
        if not dashboard:
            raise ValueError(f"Dashboard with UID '{dashboard_id}' not found.")

        dashboard_body = dashboard["dashboard"]
        panels = dashboard_body.get("panels", [])
        for idx, panel in enumerate(panels):
            if panel.get("id") == panel_id:
                self.logger.debug(f"Updating panel ID {panel_id} in dashboard '{dashboard_id}'.")
                panels[idx].update(panel_config)
                break
        else:
            raise ValueError(f"Panel with ID '{panel_id}' not found in dashboard '{dashboard_id}'.")

        dashboard_body["panels"] = panels

        # Update dashboard
        payload = {
            "dashboard": dashboard_body,
            "folderId": dashboard["meta"]["folderId"],
            "overwrite": True,
        }
        endpoint = f"{self.url}/api/dashboards/db"
        response = requests.post(endpoint, headers=self.headers, json=payload)
        if response.status_code == 200:
            self.logger.info("Panel updated successfully.")
            return response.json()
        else:
            self.logger.error(f"Failed to update panel: {response.text}")
            response.raise_for_status()

    def delete_dashboard(self, dashboard_id: str) -> None:
        """
        Delete a dashboard from Grafana.

        Args:
            dashboard_id: The UID of the dashboard to delete.
        """
        dashboard = self.get_dashboard_by_uid(dashboard_id)
        if not dashboard:
            self.logger.warning(
                f"Dashboard with UID '{dashboard_id}' not found. Nothing to delete."
            )
            return

        folder_id = dashboard["meta"]["folderId"]
        endpoint = f"{self.url}/api/dashboards/uid/{dashboard_id}"
        self.logger.debug(f"Deleting dashboard '{dashboard_id}' from folder ID {folder_id}.")
        response = requests.delete(endpoint, headers=self.headers)
        if response.status_code == 200:
            self.logger.info(f"Dashboard '{dashboard_id}' deleted successfully.")
        else:
            self.logger.error(f"Failed to delete dashboard: {response.text}")
            response.raise_for_status()

    def list_dashboards(self) -> List[Dict[str, Any]]:
        """
        Retrieve a list of all dashboards in Grafana.

        Returns:
            A list of dictionaries, each representing a dashboard.
        """
        endpoint = f"{self.url}/api/search?type=dash-db"
        self.logger.debug(f"Listing all dashboards using endpoint: {endpoint}")
        response = requests.get(endpoint, headers=self.headers)
        if response.status_code == 200:
            dashboards = response.json()
            self.logger.info(f"Retrieved {len(dashboards)} dashboards.")
            return dashboards
        else:
            self.logger.error(f"Failed to list dashboards: {response.text}")
            response.raise_for_status()

    def get_dashboard_by_uid(self, dashboard_uid: str) -> Dict[str, Any]:
        """
        Retrieve a dashboard by its UID.

        Args:
            dashboard_uid: The UID of the dashboard.

        Returns:
            A dictionary representing the dashboard, or None if not found.
        """
        endpoint = f"{self.url}/api/dashboards/uid/{dashboard_uid}"
        self.logger.debug(
            f"Retrieving dashboard with UID '{dashboard_uid}' using endpoint: {endpoint}"
        )
        response = requests.get(endpoint, headers=self.headers)
        if response.status_code == 200:
            self.logger.info(f"Dashboard '{dashboard_uid}' retrieved successfully.")
            return response.json()
        elif response.status_code == 404:
            self.logger.warning(f"Dashboard with UID '{dashboard_uid}' not found.")
            return {}
        else:
            self.logger.error(f"Failed to retrieve dashboard: {response.text}")
            response.raise_for_status()
