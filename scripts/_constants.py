################################
### Tyler Ard                ###
### Argonne National Lab     ###
### Vehicle Mobility Systems ###
### tard(at)anl(dot)gov      ###
################################

### Enums
def enum(**enums):
    return type('Enum', (), enums)

# Different traffic configurations
ROUTE = enum(NONE=0, SYNTHETIC=1, PEACHTREE=2, CONSTRUCTION=3, OFFSITE=4, SIM1=5,
             CHICAGO_2=12, CHICAGO_3=13
             ) # Enums for route scenario that codes TL placement and timings as assigned in RD matrix

GEOMETRY = enum(NONE=0, MERGE1=2, SINGLELANE=1) # Enums for road geometry scenario that codes obstacle and divider placement

TRAFFIC = enum(NONE=-1, RANDOM=0, MERGE1=1, SIM1=2, SINGLECAV=9, MIXEDRANDOM=10,
               CHICAGO_LOWER_BOUND=100
               ) # Enums for road traffic scenario that codes initial positioning and desired speed

# Different object types and statuses
LIGHTSTATUS = enum(GREEN=2, AMBER=1, RED=0) # ANL enums for light status
LIGHTTYPE = enum(LIGHT=10, STOP=20) # ANL enums for intersection type in RD matrix - traffic light or stop sign

VEHSTATUS = enum(NONE=0, FREE=1, LOWUTIL=2, TLINTAC=3) # Enums for MOBIL status on its current driving pattern
VEHTYPE = enum(SAS=-3, EXT=-2, NONE=-1, MOBIL=0, CAV=1, PCC=2, CYCLE=3) # Enums for type of vehicle as assigned in SIM matrix

### Settings
# Simulation settings
RANDSEED = 44 # Random seed used to fix the pseudo random generation. None to use system time

TEND = 205.0 # time til simulation ends automatically [s]

# Vehicle parameter settings
VEHWIDTH = 1.90 # assumed vehicle width [m]
VEHLENGTH = 3.25 # assumed vehicle length [m]

# Connectivity settings
CONN_RANGE = 450 # [m] Reliable connectivity range based on Chicago testing
SENSOR_RANGE = 100 # [m] Reliable sensor detection range

# Prediction settings
USING_PRED = False # If using prediction provided from cycle
USING_PREVIEW = False # If using a direct preview of upcoming cycle

### Error checking
assert not (USING_PRED and USING_PREVIEW), 'Cannot use both PRED and PREVIEW settings.'

if USING_PRED:
    print('Using externally-provided prediction.')
elif USING_PREVIEW:
    print('Using future preview of cycle motion.')

assert not (USING_PRED), 'USING PRED Not implemented.'
