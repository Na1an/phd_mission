# A valid file properly populated
#LASfile <- system.file("extdata", "Topography.laz", package="lidR")

las = readLAS("/home/yuchen/Documents/PhD/data_for_project/22-05-16_vol_trajet_new_vol_data_after_icp/YS-20211018-204147_bis.las")
plot(las)

# pmin = 15 because it is an extremely small file
# strongly decimated to reduce its size. There are
# actually few multiple returns
flightlines <- track_sensor(las, Roussel2020(pmin = 15))

plot(las@header)
plot(sf::st_geometry(flightlines), add = TRUE)

#plot(las) |> add_flightlines3d(flightlines, radius = 10)

## Not run: 
# With a LAScatalog "-drop_single" and "-thin_pulses_with_time"
# are used by default
ctg = readLAScatalog("folder/")
flightlines <- track_sensor(ctg,  Roussel2020(pmin = 15))
plot(flightlines)

## End(Not run)