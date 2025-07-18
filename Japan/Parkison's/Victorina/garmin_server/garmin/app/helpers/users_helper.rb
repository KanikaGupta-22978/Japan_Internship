module UsersHelper
	def interpret_stress(stress_score)
		case stress_score
		when -2
			"Too much motion"
		when -1..0
			"Not enough data"
		when 1..25
			"Rest"
		when 26..50
			"Low"
		when 51..75
			"Medium"
		when 76..100
			"High"
		end
	end
end
