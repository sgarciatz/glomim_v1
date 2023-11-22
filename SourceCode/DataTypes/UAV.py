from .Microservice import Microservice

class UAV(object):

    def __init__(self, uavId: str, position: list[int],
                 ramCapacity: float = 4.0, ramAllocated: float = 0.0,
                 cpuCapacity: float = 4.0, cpuAllocated: float = 0.0,
                 microservices: list[Microservice] = [] ) -> None:

        self.__id            = uavId
        self.__position      = position
        self.__ramCapacity   = ramCapacity
        self.__ramAllocated  = ramAllocated
        self.__cpuCapacity   = cpuCapacity
        self.__cpuAllocated  = cpuAllocated
        self.__microservices = microservices

    @property
    def id(self) -> str:
        return self.__id

    @id.setter
    def id(self, uavId: str) -> None:
        self.__id = uavId

    @property
    def position(self) -> list[int]:
        return self.__position

    @position.setter
    def position(self, newPosition: list[int]) -> None:
        self.__position = newPosition

    @property
    def ramCapacity(self) -> float:
        return self.__ramCapacity

    @ramCapacity.setter
    def ramCapacity(self, newRamCapacity: float) -> None:
        self.__ramCapacity = newRamCapacity

    @property
    def ramAllocated(self) -> float:
        return self.__ramAllocated

    @ramAllocated.setter
    def ramAllocated(self, newRamAllocated: float) -> None:
        if (newRamAllocated > self.__ramCapacity):
            raise Exception(f'Can not allocate {newRamAllocated} of RAM for UAV {id} with {self.__ramCapacity} of RAM capacity')
        self.__ramAllocated = newRamAllocated

    @property
    def cpuCapacity(self) -> float:
        return self.__cpuCapacity

    @cpuCapacity.setter
    def cpuCapacity(self, newCpuCapacity: float) -> None:
        self.__cpuCapacity = newCpuCapacity

    @property
    def cpuAllocated(self) -> float:
        return self.__cpuAllocated

    @cpuAllocated.setter
    def cpuAllocated(self, newCpuAllocated) -> float:
        if (newCpuAllocated > self.__cpuCapacity):
            raise Exception(f'Can not allocate {newCpuAllocated} of CPU for UAV {id} with {self.__cpuCapacity} of CPU capacity')
        self.__cpuAllocated = newCpuAllocated

    @property
    def microservices(self) -> list[Microservice]:
        return self.__microservices

    @microservices.setter
    def microservices(self, newMicroservices: list[Microservice]) -> None:
        self.__microservices = newMicroservices

    def deployMicroservice(self, ms: Microservice) -> bool:

        """Deploy a microservice if it fits"""

        ramRemaining: float = \
            self.__ramCapacity - (self.__ramAllocated + ms.ramRequirement)
        cpuRemaining: float = \
            self.__cpuCapacity - (self.__cpuAllocated + ms.cpuRequirement)
        if (ramRemaining < 0):
            return False
        if (cpuRemaining < 0):
            return False
        self.__ramAllocated += ms.ramRequirement
        self.__cpuAllocated += ms.cpuRequirement
        self.__microservices.append(ms)
        return True 
            
    def __str__(self) -> str:
        microservicesString = ''
        for ms in self.__microservices:
            microservicesString += f'{ms.id} '
        return f'UAV id: {self.__id}\n\t-Position: {self.__position}\n\t-RAM: {self.__ramCapacity} (capacity) {self.__ramAllocated} (allocated) \n\t-CPU: {self.__cpuCapacity} (capacity) {self.__cpuAllocated} (allocated)\n\t-Microservices: {microservicesString}'
        
    def toJSON(self) -> dict:
        json = {
            'uavId'         : self.__id,
            'position'      : self.__position,
            'ramCapacity'   : self.__ramCapacity,
            'ramAllocated'  : self.__ramAllocated,
            'cpuCapacity'   : self.__cpuCapacity,
            'cpuAllocated'  : self.__cpuAllocated,
            'microservices' : [ ms for ms in self.__microservices]
            }
        return json
