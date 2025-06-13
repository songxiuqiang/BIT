#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#include "mbt/m_func.hh"

namespace summer
{

#define SIGN(x) (((x) >= 1e-12) ? 1 : -1)

	/* L2 */
	float l2_cost(float x, float c)
	{
		return 0.5 * x * x;
	}
	float l2_influence(float x, float c)
	{
		return x;
	}
	float l2_weight(float x, float c)
	{
		return 1;
	}

	/* L1 */
	float l1_cost(float x, float c)
	{
		return fabs(x);
	}
	float l1_influence(float x, float c)
	{
		return SIGN(x);
	}
	float l1_weight(float x, float c)
	{
		return 1 / SIGN(x);
	}

	/* L1L2 */
	float l1l2_cost(float x, float c)
	{
		return 2 * (sqrtf((1 + 0.5 * x * x)) - 1);
	}
	float l1l2_influence(float x, float c)
	{
		return x / sqrtf((1 + 0.5 * x * x));
	}
	float l1l2_weight(float x, float c)
	{
		return 1 / sqrtf((1 + 0.5 * x * x));
	}

	/* Fair */
	float fair_cost(float x, float c)
	{
		return c * c * (fabs(x) / c - logf((1 + fabs(x) / c)));
	}
	float fair_influence(float x, float c)
	{
		return x / (1 + fabs(x) / c);
	}
	float fair_weight(float x, float c)
	{
		return 1 / (1 + fabs(x) / c);
	}

	/* Huber */
	float huber_cost(float x, float c)
	{
		if (fabs(x) <= c)
		{
			return 0.5 * x * x;
		}
		else
		{
			return c * (fabs(x) - 0.5 * c);
		}
	}
	float huber_influence(float x, float c)
	{
		if (fabs(x) <= c)
		{
			return x;
		}
		else
		{
			return c * SIGN(x);
		}
	}
	float huber_weight(float x, float c)
	{
		if (fabs(x) <= c)
		{
			return 1;
		}
		else
		{
			return c / fabs(x);
		}
	}

	/* CAUCHY */
	float cauchy_cost(float x, float c)
	{
		return 0.5 * c * c * logf((1 + (x / c) * (x / c)));
	}
	float cauchy_influence(float x, float c)
	{
		return x / (1 + (x / c) * (x / c));
	}
	float cauchy_weight(float x, float c)
	{
		return 1 / (1 + (x / c) * (x / c));
	}

	/* GEMANMCCLURE */
	float gemanmcclure_cost(float x, float c)
	{
		return 0.5 * x * x / (c + x * x);
	}
	float gemanmcclure_influence(float x, float c)
	{
		return x / ((c + x * x) * (c + x * x));
	}
	float gemanmcclure_weight(float x, float c)
	{
		return 1 / ((c + x * x) * (c + x * x));
	}

	/* WELSCH */
	float welsch_cost(float x, float c)
	{
		return 0.5 * c * c * (1 - expf((-(x / c) * (x / c))));
	}
	float welsch_influence(float x, float c)
	{
		return x * expf((-(x / c) * (x / c)));
	}
	float welsch_weight(float x, float c)
	{
		return expf((-(x / c) * (x / c)));
	}

	/* TUKEY */
	float tukey_cost(float x, float c)
	{
		if (fabs(x) <= c)
		{
			return (c * c / 6) * (1 - (1 - (x / c) * (x / c)) * (1 - (x / c) * (x / c)) * (1 - (x / c) * (x / c)));
		}
		else
		{
			return c * c / 6;
		}
	}
	float tukey_influence(float x, float c)
	{
		if (fabs(x) <= c)
		{
			return x * (1 - (x / c) * (x / c)) * (1 - (x / c) * (x / c));
		}
		else
		{
			return 0;
		}
	}
	float tukey_weight(float x, float c)
	{
		if (fabs(x) <= c)
		{
			return (1 - (x / c) * (x / c)) * (1 - (x / c) * (x / c));
		}
		else
		{
			return 0;
		}
	}

} // ns slot