using WAV

sounds_source = ["game_over", "stage_clear", "coin", "gangnamc", "yagaga", "heeeh"]

sounds = Dict(
  (key, wavread("sounds/$key.wav")) for key in sounds_source
)

function beep(sound_names...)
  for sound_name in sound_names
    wavplay(sounds[sound_name][1], sounds[sound_name][2])
  end
end

macro beep(expression)
  return quote
    try
      result = $(esc(expression))
      beep("gangnamc", "coin")
      result
    catch e
      beep("game_over")
      throw(e)
    end
  end
end