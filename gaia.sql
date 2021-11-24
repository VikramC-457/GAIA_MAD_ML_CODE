SELECT bp_rp, parallax, pmra, pmdec, phot_g_mean_mag AS gp
FROM gaiadr2.gaia_source
WHERE 1 = CONTAINS(POINT('ICRS', ra, dec),
                   CIRCLE('ICRS', 315.4042, 68.1631, 1))
AND phot_g_mean_flux_over_error > 50
AND phot_rp_mean_flux_over_error > 20
AND phot_bp_mean_flux_over_error > 20
AND phot_bp_rp_excess_factor < 1.3 + 0.06 * POW(bp_rp, 2)
AND phot_bp_rp_excess_factor > 1.0 + 0.015 * POW(bp_rp, 2)
AND visibility_periods_used > 8
