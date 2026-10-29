"""Batch vehicle observations without changing SUMO steps or control logic."""

from functools import partial
import traci.constants as tc


class VehicleCache:
    VARIABLES = {
        "getSpeed": tc.VAR_SPEED,
        "getAllowedSpeed": tc.VAR_ALLOWED_SPEED,
        "getCO2Emission": tc.VAR_CO2EMISSION,
        "getMinGap": tc.VAR_MINGAP,
        "getLaneID": tc.VAR_LANE_ID,
        "getPosition": tc.VAR_POSITION,
        "getAngle": tc.VAR_ANGLE,
    }

    def __init__(self, domain):
        self.domain = domain
        self.subscribed = set()
        self.values = {}
        self.ids = None

    def refresh(self):
        self.ids = self.domain.getIDList()
        active = set(self.ids)
        self.subscribed.intersection_update(active)
        for vehicle in self.ids:
            if vehicle not in self.subscribed:
                self.domain.subscribe(
                    vehicle, [*self.VARIABLES.values(), tc.VAR_LEADER],
                    parameters={tc.VAR_LEADER: ("d", 100.)})
                self.subscribed.add(vehicle)
        self.values = self.domain.getAllSubscriptionResults()

    def _get(self, method, vehicle):
        variable = self.VARIABLES[method]
        values = self.values.get(vehicle, {})
        if variable in values:
            return values[variable]
        return getattr(self.domain, method)(vehicle)

    def getLeader(self, vehicle, dist=100.):
        values = self.values.get(vehicle, {})
        if dist == 100. and tc.VAR_LEADER in values:
            return values[tc.VAR_LEADER]
        return self.domain.getLeader(vehicle, dist)

    def getIDList(self):
        return self.ids if self.ids is not None else self.domain.getIDList()

    def __getattr__(self, name):
        if name in self.VARIABLES:
            return partial(self._get, name)
        return getattr(self.domain, name)


class CachedTraCI:
    def __init__(self, connection):
        self.connection = connection
        self.vehicle = VehicleCache(connection.vehicle)

    def start(self, *args, **kwargs):
        result = self.connection.start(*args, **kwargs)
        self.vehicle = VehicleCache(self.connection.vehicle)
        return result

    def simulationStep(self, *args, **kwargs):
        result = self.connection.simulationStep(*args, **kwargs)
        self.vehicle.refresh()
        return result

    def __getattr__(self, name):
        return getattr(self.connection, name)
